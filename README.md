# RAG Movie Recommender (Gemini + Supabase)

A simple movie recommender that:
- Ingests movies, stores Gemini embeddings in Supabase (pgvector)
- Returns recommendations from a questionnaire, with optional LLM rerank/reasons
- Ships with a minimal Tailwind HTML frontend

## Stack
- Backend: Node.js + TypeScript + Express
- Vector DB: Supabase (Postgres + pgvector)
- Embeddings/LLM: Google Gemini (`text-embedding-004`, `gemini-1.5-flash`)
- Frontend: Static HTML + Tailwind (served by Express)

## Features
- Bulk ingest movies with precomputed embeddings
- Recommend top-N by cosine similarity (pgvector)
- Optional LLM-based rerank with short “why” reasons
- Simple web UI for ingest + recommend

## Quick Start

### 1) Install
```bash
npm i
# or explicitly:
npm i express zod dotenv @supabase/supabase-js pino pino-pretty cors @google/generative-ai
npm i -D @types/node
```

### 2) Environment
Create `.env` at project root:
```bash
PORT=3000
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
SUPABASE_URL=YOUR_SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY=YOUR_SUPABASE_SERVICE_ROLE_KEY
EMBED_MODEL=text-embedding-004
VECTOR_DIM=768
TOP_N=5
SIM_THRESHOLD=0.25
```

Optional `.env.example` is provided in this README for reference—do not commit real secrets.

### 3) Supabase SQL (run once)
Enable vector extension, create table, index, and RPC:
```sql
create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists public.movies (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  year int,
  genres text[] default array[]::text[],
  cast text[] default array[]::text[],
  plot text default '',
  metadata jsonb default '{}',
  embedding vector(768)  -- matches text-embedding-004
);

create index if not exists movies_embedding_idx
  on public.movies using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

create index if not exists movies_title_idx on public.movies (title);

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
    m.id, m.title, m.year, m.genres, m.plot, m.metadata,
    1 - (m.embedding <=> query_embedding) as score
  from public.movies m
  where 1 - (m.embedding <=> query_embedding) >= sim_threshold
  order by m.embedding <=> query_embedding
  limit match_count;
$$;
```

### 4) Run
```bash
# dev (ts-node)
npx ts-node src/server.ts

# or build + run
npx tsc
node dist/server.js
```

Open http://localhost:3000 — the frontend is served from `web/`.

## Frontend
- `web/index.html` provides two panels:
  - Ingest: paste a JSON array of movies
  - Recommend: enter answers (one per line), optional rerank toggle
- The HTML uses `tailwindcss.com` CDN and calls `/api/ingest/movies` and `/api/recommend`.

## API

### Health
- GET `/health`
- Response: `{ "ok": true }`

### Ingest Movies
- POST `/api/ingest/movies`
- Body:
```json
{
  "movies": [
    {
      "title": "Inception",
      "year": 2010,
      "genres": ["Sci-Fi", "Thriller"],
      "cast": ["Leonardo DiCaprio"],
      "plot": "A thief enters dreams to plant ideas.",
      "metadata": { "director": "Christopher Nolan" }
    }
  ]
}
```
- Notes:
  - Omit `id` to let DB auto-generate.
  - `embedding` is computed server-side using Gemini.
- Response: `{ "inserted": 1, "ids": ["..."] }`

### Recommend
- POST `/api/recommend`
- Body:
```json
{
  "answers": [
    "I want something mind-bending and thought-provoking",
    "Prefer sci-fi or mystery with strong plot twists",
    "I enjoyed Interstellar and Shutter Island",
    "Not too much gore, prefer emotional depth",
    "Ideally from the last 15 years"
  ],
  "topN": 5,
  "offset": 0,
  "threshold": 0.25,
  "rerank": false
}
```
- Response:
```json
{
  "topN": [
    {
      "id": "...",
      "title": "Inception",
      "year": 2010,
      "genres": ["Sci-Fi", "Thriller"],
      "score": 0.91,
      "why": "Optional short reason if rerank=true",
      "metadata": { "director": "Christopher Nolan" }
    }
  ]
}
```

## Sample Data
Use this JSON to test ingest:
```json
[
  {
    "title": "Inception",
    "year": 2010,
    "genres": ["Sci-Fi", "Thriller"],
    "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"],
    "plot": "A thief who steals corporate secrets through dream-sharing is given an inverse task: plant an idea.",
    "metadata": { "director": "Christopher Nolan", "country": "USA" }
  },
  {
    "title": "La La Land",
    "year": 2016,
    "genres": ["Romance", "Musical"],
    "cast": ["Emma Stone", "Ryan Gosling"],
    "plot": "A jazz pianist and an aspiring actress navigate love and ambition in Los Angeles.",
    "metadata": { "director": "Damien Chazelle", "awards": ["Academy Awards"] }
  }
]
```

## Why VECTOR_DIM=768?
`text-embedding-004` (Gemini) outputs 768-dim vectors. Your DB column, RPC function, and `VECTOR_DIM` must match that size—or inserts/queries will fail.

## Notes & Tips
- Only include `id` on ingest when you actually have one; otherwise omit so Postgres can generate `gen_random_uuid()`.
- If you accidentally tracked secrets, rotate them and remove from history (e.g., `git filter-repo` or BFG).
- Tuning:
  - Increase `SIM_THRESHOLD` for stricter matches; lower for more results.
  - Enrich embeddings with more metadata (keywords, critic quotes, tags).
  - Rerank is optional; keep off for latency-sensitive paths.

## Scripts (optional)
Add to `package.json`:
```json
{
  "scripts": {
    "dev": "ts-node src/server.ts",
    "build": "tsc",
    "start": "node dist/server.js"
  }
}
```

## License
MIT

