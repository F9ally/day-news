-- Monthly Summary feature: Supabase schema & policy setup
-- Run these statements in the Supabase SQL editor (or via psql) for your project.
--
-- This creates the `monthly_digests` table used by scripts/generate_monthly_summary.py
-- and day-news-mobile's Monthly Summary screen, mirroring `daily_digests` but keyed
-- by calendar month (YYYY-MM) instead of day.

BEGIN;

CREATE TABLE IF NOT EXISTS public.monthly_digests (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  month text NOT NULL, -- format: 'YYYY-MM'
  compiled text NOT NULL,
  items jsonb NOT NULL DEFAULT '[]'::jsonb
);

-- Required for the ON CONFLICT (merge-duplicates) upsert used by the publisher script.
CREATE UNIQUE INDEX IF NOT EXISTS monthly_digests_month_key
  ON public.monthly_digests (month);

-- Enable Row Level Security and allow public (anon) read access, matching daily_digests.
ALTER TABLE public.monthly_digests ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS anon_select_monthly_digests ON public.monthly_digests;
CREATE POLICY anon_select_monthly_digests
  ON public.monthly_digests
  FOR SELECT
  TO anon
  USING (true);

-- The publisher script uses the service_role key (bypasses RLS) for inserts/upserts,
-- so no INSERT/UPDATE policy for `anon` is required or granted.

COMMIT;

-- ---------------------------------------------------------------------------
-- Storage: monthly audio narration reuses the existing public `news-audio`
-- bucket (created automatically by the daily digest script) with filenames
-- like `monthly-2026-06.mp3`. No additional bucket or policy changes are
-- needed as long as that bucket is public, which the publisher scripts
-- already ensure via ensure_supabase_bucket().
--
-- To verify/create it manually instead:
--   insert into storage.buckets (id, name, public)
--   values ('news-audio', 'news-audio', true)
--   on conflict (id) do update set public = true;
-- ---------------------------------------------------------------------------

-- Sanity check: list policies afterwards
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual AS using_expr
FROM pg_policies
WHERE schemaname = 'public' AND tablename = 'monthly_digests';
