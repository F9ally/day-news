-- Inspect current RLS policies on the table
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual AS using_expr, with_check
FROM pg_policies
WHERE schemaname = 'public' AND tablename = 'daily_digests';

-- If there are multiple permissive anon SELECT policies, create a single unified policy.
-- This version allows public read of all rows (adjust USING(...) if you need restrictions).
BEGIN;
  -- Ensure RLS is enabled (no-op if already enabled)
  ALTER TABLE public.daily_digests ENABLE ROW LEVEL SECURITY;

  -- Create a new unified anon SELECT policy; pick a unique name
  CREATE POLICY anon_select_unified
  ON public.daily_digests
  FOR SELECT
  TO anon
  USING (true);

  -- Drop the older duplicate policies (ignore if they don't exist)
  DROP POLICY IF EXISTS "Allow select for anon" ON public.daily_digests;
  DROP POLICY IF EXISTS anon_select ON public.daily_digests;
COMMIT;

-- OPTIONAL: If you need to combine existing conditions rather than public read,
-- query current policies' expressions and OR them manually into the unified policy.
-- Example template:
-- BEGIN;
--   CREATE POLICY anon_select_unified
--   ON public.daily_digests
--   FOR SELECT
--   TO anon
--   USING ( /* cond_from_policy_A */ OR /* cond_from_policy_B */ );
--   DROP POLICY IF EXISTS "Allow select for anon" ON public.daily_digests;
--   DROP POLICY IF EXISTS anon_select ON public.daily_digests;
-- COMMIT;
