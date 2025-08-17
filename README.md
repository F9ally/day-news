# day-news

Simple static site plus an automation script to fetch news, summarize with local Mistral 7B (via Ollama), and publish a daily digest to Supabase.

## Automation script: fetch, summarize, and publish digest

This workspace includes a Python script that:

- Fetches 1 recent news article per topic (10 topics)
- Summarizes each using local Ollama Mistral (7B) if available
- Compiles a single numbered digest
- Inserts the digest into a Supabase table (`daily_digests` by default)

Files:
- `scripts/fetch_and_publish_digest.py` – main script
- `requirements.txt` – Python dependencies
- `ven.env` – Provide `NEWS_API_URL`, `NEWS_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`. Optional: `OLLAMA_MODEL` (default `mistral`), `SUPABASE_TABLE` (default `daily_digests`).

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

### Run locally

1) Ensure Ollama is installed and `mistral` model is available:
	- `ollama pull mistral` (or set `OLLAMA_MODEL` in `ven.env`)

2) Install Python deps:
	- `pip install -r requirements.txt`

3) Execute the script:
	- `python scripts/fetch_and_publish_digest.py`

The script saves a local copy under `out/` even if Supabase insert fails.