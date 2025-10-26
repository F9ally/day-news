# day-news

Simple static site plus an automation script to fetch news, summarize with local Gemma3:4b (via Ollama), and publish a daily digest to Supabase.

## Automation script: fetch, summarize, and publish digest

This workspace includes a Python script that:

- Fetches 1 recent news article per topic (9 topics)
- Summarizes each using local Ollama Gemma3:4b if available
- Compiles a single numbered digest
- Inserts the digest into a Supabase table (`daily_digests` by default)

Files:
- `scripts/fetch_and_publish_digest.py` – main script
- `requirements.txt` – Python dependencies
- `ven.env` – Provide `NEWS_API_URL`, `NEWS_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`. Optional: `NEWS_LANG`, `NEWS_COUNTRY`, `NEWS_MAX`, `NEWS_NULLABLE`, `NEWS_QUERY`, `NEWS_FROM`, `NEWS_TO`, `OLLAMA_MODEL` (default `mistral`), `SUPABASE_TABLE` (default `daily_digests`).

Expected Supabase table schema (SQL):

```
create table if not exists public.daily_digests (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz default now(),
  date date,
  compiled text,
  items jsonb
);
```
