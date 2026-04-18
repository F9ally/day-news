/**
 * UTC date helpers — mirrors the website's time logic exactly.
 * Before 06:25 UTC → show yesterday's digest.
 * After 06:25 UTC  → show today's digest.
 */

export function getUTCDateStr(d: Date = new Date()): string {
  return d.toISOString().slice(0, 10);
}

export function getYesterdayUTCDateStr(): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - 1);
  return getUTCDateStr(d);
}

export function isBeforeSixTwentyFiveUTC(d: Date = new Date()): boolean {
  const h = d.getUTCHours();
  const m = d.getUTCMinutes();
  return h < 6 || (h === 6 && m < 25);
}

export function getDisplayDateUTC(): string {
  return isBeforeSixTwentyFiveUTC() ? getYesterdayUTCDateStr() : getUTCDateStr();
}

export function formatDateForDisplay(dateStr: string): string {
  const date = new Date(dateStr + 'T00:00:00Z');
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    timeZone: 'UTC',
  });
}

export function msUntilNextUTCMidnight(): number {
  const now = new Date();
  const next = Date.UTC(
    now.getUTCFullYear(),
    now.getUTCMonth(),
    now.getUTCDate() + 1,
    0, 0, 0, 0,
  );
  return next - now.getTime();
}

export function msUntilNextUTC625(): number {
  const now = new Date();
  const today625 = Date.UTC(
    now.getUTCFullYear(),
    now.getUTCMonth(),
    now.getUTCDate(),
    6, 25, 0, 0,
  );
  const target =
    now.getTime() < today625
      ? today625
      : Date.UTC(
          now.getUTCFullYear(),
          now.getUTCMonth(),
          now.getUTCDate() + 1,
          6, 25, 0, 0,
        );
  return target - now.getTime();
}

/**
 * Advance or rewind a YYYY-MM-DD date by `days`.
 */
export function shiftDate(dateStr: string, days: number): string {
  const d = new Date(dateStr + 'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() + days);
  return getUTCDateStr(d);
}
