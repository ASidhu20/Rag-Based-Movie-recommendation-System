/*
RAG Movie Recommender — Backend (API + DB + Embeddings)
Stack: Node.js + TypeScript + Express, Supabase (Postgres + pgvector), Gemini Embeddings

This single file shows:
1) API server (Express)
2) Supabase + Gemini clients
3) Endpoints: /health, /api/ingest/movies, /api/recommend
4) Data validation
5) SQL you must run in Supabase (at bottom of file in a multi-line comment)

How to run:
- npm i express zod dotenv @supabase/supabase-js pino pino-pretty cors @google/generative-ai
- npm i -D @types/node
- ts-node src/server.ts  (or compile with tsc then node dist/server.js)

ENV required (.env):
PORT=3000
GOOGLE_API_KEY=...       # Gemini API key
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...   # service key required for RPC/insert with RLS
EMBED_MODEL=text-embedding-004
TOP_N=5
SIM_THRESHOLD=0.25               # 0..1 (cosine-sim). 0.25 is a good starting point
VECTOR_DIM=768                   # dim for text-embedding-004
*/

import 'dotenv/config'
import express, { Request, Response } from 'express'
import cors from 'cors'
import { z } from 'zod'
import { createClient } from '@supabase/supabase-js'
import pino from 'pino'
import { GoogleGenerativeAI } from '@google/generative-ai'

// // --- Setup --------------------------------------------------------------
// const app = express()
// app.use(express.json({ limit: '2mb' }))
// app.use(cors())
// const log = pino({ transport: { target: 'pino-pretty' } })

// const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || '')

// --- Setup --------------------------------------------------------------
const app = express()
app.use(express.json({ limit: '2mb' }))
app.use(cors())

// Serve the frontend (web/index.html and assets) from the same app
app.use(express.static('web'))

const log = pino({ transport: { target: 'pino-pretty' } })

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || '')

const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
  { auth: { persistSession: false } }
)

const EMBED_MODEL = process.env.EMBED_MODEL || 'text-embedding-004'
const VECTOR_DIM = Number(process.env.VECTOR_DIM || 768)
const DEFAULT_TOP_N = Number(process.env.TOP_N || 5)
const DEFAULT_SIM_THRESHOLD = Number(process.env.SIM_THRESHOLD || 0.25)

// --- Schemas ------------------------------------------------------------
const MovieSchema = z.object({
  title: z.string(),
  year: z.number().int().optional(),
  genres: z.array(z.string()).default([]),
  cast: z.array(z.string()).default([]),
  plot: z.string().default(''),
  metadata: z.record(z.string(), z.any()).default({})
})

const IngestBodySchema = z.object({ movies: z.array(MovieSchema).min(1) })

const RecommendBodySchema = z.object({
  answers: z.array(z.string()).min(1).max(10),
  topN: z.number().int().min(1).max(50).optional(),
  offset: z.number().int().min(0).max(500).optional(),
  threshold: z.number().min(0).max(1).optional(),
  // Optional rerank toggle (LLM reasoning on short-list) — not required to start
  rerank: z.boolean().optional()
})

// --- Helpers ------------------------------------------------------------
function joinAnswers(answers: string[]) {
  // Join user answers into one preference profile
  return [
    'You are building a movie preference profile from questionnaire answers.',
    'Summarize consistent tastes implicitly. Answers:',
    ...answers.map((a, i) => `Q${i + 1}: ${a}`)
  ].join('\n')
}

function movieTextForEmbedding(m: z.infer<typeof MovieSchema>): string {
  return [
    `Title: ${m.title}`,
    m.year ? `Year: ${m.year}` : '',
    m.genres.length ? `Genres: ${m.genres.join(', ')}` : '',
    m.cast.length ? `Cast: ${m.cast.join(', ')}` : '',
    m.plot ? `Plot: ${m.plot}` : '',
    m.metadata ? `Metadata: ${JSON.stringify(m.metadata)}` : ''
  ].filter(Boolean).join('\n')
}

async function embed(text: string): Promise<number[]> {
  const model = genAI.getGenerativeModel({ model: EMBED_MODEL })
  const resp = await model.embedContent(text)
  return resp.embedding.values
}

// --- Routes -------------------------------------------------------------
app.get('/health', (_req: Request, res: Response) => res.json({ ok: true }))

// Bulk ingest or upsert movies with embeddings
app.post('/api/ingest/movies', async (req: Request, res: Response) => {
  const parsed = IngestBodySchema.safeParse(req.body)
  if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() })

  const movies = parsed.data.movies

  try {
    // Compute embeddings in sequence to keep it simple (you may parallelize with Promise.allSettled)
    const rows = [] as any[]
    for (const m of movies) {
      const content = movieTextForEmbedding(m)
      const vector = await embed(content)
      const row: any = {
        title: m.title, year: m.year ?? null,
        genres: m.genres, cast: m.cast, plot: m.plot,
        metadata: m.metadata, embedding: vector
      }
      // if (m.id) row.id = m.id 
      rows.push(row)
    }

    const { data, error } = await supabase.from('movies').upsert(rows).select('id, title')
    if (error) throw error

    return res.json({ inserted: data?.length ?? 0, ids: data?.map(d => d.id) })
  } catch (e: any) {
    log.error(e)
    return res.status(500).json({ error: e.message })
  }
})

// Recommend top N movies from questionnaire answers
app.post('/api/recommend', async (req: Request, res: Response) => {
  const parsed = RecommendBodySchema.safeParse(req.body)
  if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() })

  const { answers, rerank = false } = parsed.data
  const topN = parsed.data.topN ?? DEFAULT_TOP_N
  const offset = parsed.data.offset ?? 0
  const threshold = parsed.data.threshold ?? DEFAULT_SIM_THRESHOLD

  try {
    const profile = joinAnswers(answers)
    const qvec = await embed(profile)

    // Call RPC that orders by cosine distance & filters by similarity threshold
    const { data, error } = await supabase.rpc('match_movies', {
      query_embedding: qvec,
      match_count: topN + offset,
      sim_threshold: threshold
    })
    if (error) throw error

    // Offset/paginate on the app side
    const sliced = (data as any[]).slice(offset, offset + topN)

    // Optional: rerank short-list with a quick reasoner (kept off by default)
    if (!rerank) {
      return res.json({
        topN: sliced.map((r: any) => ({
          id: r.id,
          title: r.title,
          year: r.year,
          genres: r.genres,
          score: r.score, // cosine similarity 0..1
          why: r.why ?? null,
          metadata: r.metadata
        }))
      })
    }

    // Lightweight rerank prompt (kept brief; deterministic-ish)
    const listForLLM = sliced
      .map((r: any, i: number) => `${i + 1}. ${r.title} (${r.year}) | genres=${r.genres?.join(', ')}`)
      .join('\n')

    const rerankPrompt = [
      'Rank movies 1..N for best match to the given preference profile.',
      'Respond ONLY as JSON array of {title, reason} in best-to-worst order.',
      '',
      `Preference profile (from answers):\n${profile}`,
      '',
      `Candidates:\n${listForLLM}`
    ].join('\n')

    const gpt = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' })
    const chat = await gpt.generateContent(rerankPrompt)
    const text = chat.response.text()

    let reranked: any[] = []
    try { reranked = JSON.parse(text || '[]') } catch {}

    // Merge LLM reasons back onto results (preserving original scores)
    const withReason = sliced.map((r: any) => {
      const found = reranked.find((x: any) => x.title?.toLowerCase() === r.title?.toLowerCase())
      return { ...r, why: found?.reason ?? null }
    })

    return res.json({ topN: withReason })
  } catch (e: any) {
    log.error(e)
    return res.status(500).json({ error: e.message })
  }
})

// --- Start --------------------------------------------------------------
const port = Number(process.env.PORT || 3000)
app.listen(port, () => log.info(`API listening on http://localhost:${port}`))

/*
==================== Supabase SQL (run once) ====================
-- Enable extension
create extension if not exists vector;

-- Movies table (embedding dim must match your model)
create table if not exists public.movies (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  year int,
  genres text[] default array[]::text[],
  cast text[] default array[]::text[],
  plot text default '',
  metadata jsonb default '{}',
  embedding vector(768)  -- align with VECTOR_DIM/embedding model (text-embedding-004)
);

-- Vector index (IVFFLAT with cosine distance). Adjust lists for data size.
create index if not exists movies_embedding_idx
  on public.movies using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Helpful btree on title for quick admin lookups
create index if not exists movies_title_idx on public.movies (title);

-- RPC to match by cosine similarity, return a similarity score 0..1
create or replace function public.match_movies(
  query_embedding vector(768),
  match_count int default 5,
  sim_threshold float default 0.25
)
returns table (
  id uuid,
  title text,
  year int,
  genres text[],
  plot text,
  metadata jsonb,
  score float
)
language sql stable as $$
  select
    m.id,
    m.title,
    m.year,
    m.genres,
    m.plot,
    m.metadata,
    1 - (m.embedding <=> query_embedding) as score  -- cosine similarity
  from public.movies m
  where 1 - (m.embedding <=> query_embedding) >= sim_threshold
  order by m.embedding <=> query_embedding  -- ascending distance
  limit match_count;
$$;

-- (Optional) RLS policies (if you enable RLS). Example: allow read to anon, write via service role only.
-- alter table public.movies enable row level security;
-- create policy "read_movies" on public.movies for select using (true);
-- create policy "no_direct_writes" on public.movies for insert with check (false);
-- create policy "no_updates" on public.movies for update using (false);

==================== Example Requests ====================
# Ingest a batch
curl -X POST http://localhost:3000/api/ingest/movies \
  -H 'content-type: application/json' \
  -d '{
    "movies": [
      {"title":"Inception","year":2010,"genres":["Sci-Fi","Thriller"],"cast":["Leonardo DiCaprio"],"plot":"A thief enters dreams to plant ideas."},
      {"title":"La La Land","year":2016,"genres":["Romance","Musical"],"cast":["Emma Stone","Ryan Gosling"],"plot":"A jazz pianist and an actress fall in love in LA."}
    ]
  }'

# Get recommendations (top 5 by default)
curl -X POST http://localhost:3000/api/recommend \
  -H 'content-type: application/json' \
  -d '{
    "answers": [
      "I want something mind-bending and thought-provoking",
      "Prefer sci-fi or mystery with strong plot twists",
      "I enjoyed Interstellar and Shutter Island",
      "Not too much gore, prefer emotional depth",
      "Ideally from the last 15 years"
    ]
  }'

# Pagination / show more (offset = 5 returns next 5)
curl -X POST http://localhost:3000/api/recommend \
  -H 'content-type: application/json' \
  -d '{
    "answers": ["same answers..."],
    "topN": 5,
    "offset": 5
  }'

==================== Notes & Tuning ====================
- Accuracy target (~50%):
  * Tune SIM_THRESHOLD to balance precision/recall. Higher = stricter matches.
  * Prefer higher-quality embeddings when available.
  * Enrich movie embeddings with: plot + genres + tags + keywords + critic quotes.
  * Normalize text (lowercase, strip punctuation) consistently on ingest and query.

- Cold start: If the DB is small, reduce lists in the IVFFLAT index or use a plain scan temporarily.

- Rerank: Keep rerank: false by default; enable for users who want explanations.

- Safety: Put RLS on for prod; use the Service Role key only server-side.

- Observability: add timing logs around embedding+RPC to spot bottlenecks.
*/