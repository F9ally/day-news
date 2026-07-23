"""
scripts/generate_monthly_summary.py

Aggregates a full calendar month of daily digests (from the `daily_digests`
Supabase table) into one structured "Monthly Summary": one entry per topic,
each a paragraph synthesizing that topic's most important stories for the
month, with duplicate/overlapping stories merged rather than repeated.

Output is published to a dedicated `monthly_digests` Supabase table (see
scripts/monthly_digests_schema.sql for the required schema/RLS setup) and,
like the daily digest, a spoken-audio narration is generated and uploaded to
the `news-audio` storage bucket under the key `monthly-<YYYY-MM>.mp3`.

Usage:
    python scripts/generate_monthly_summary.py
        Summarizes the most recently completed calendar month (i.e. if run
        on/after the 1st of a new month, it summarizes the previous month).

    python scripts/generate_monthly_summary.py --month 2026-06
        Summarizes an explicit month (YYYY-MM). Useful for backfilling or
        re-running a month.

Intended to be triggered once per month (e.g. via cron on the 1st, shortly
after the last daily digest of the prior month has been published).
"""
import argparse
import calendar
import io
import os
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from pydub import AudioSegment

from fetch_and_publish_digest import (  # noqa: E402
    load_config,
    compile_digest,
    build_narration_text_from_compiled,
    generate_audio_with_kokoro,
    ensure_supabase_bucket,
    upload_audio_to_supabase,
    ensure_ollama_ready,
)

MONTHLY_TABLE_DEFAULT = "monthly_digests"
AUDIO_BUCKET = "news-audio"


# --- Month selection ---------------------------------------------------------

def previous_month_str(today: Optional[date] = None) -> str:
    """Return the YYYY-MM string for the calendar month immediately before `today`."""
    d = today or datetime.now(timezone.utc).date()
    first_of_this_month = d.replace(day=1)
    last_of_prev_month = first_of_this_month.fromordinal(first_of_this_month.toordinal() - 1)
    return last_of_prev_month.strftime("%Y-%m")


def month_bounds(month_str: str) -> Tuple[str, str]:
    """Return (first_day, last_day) ISO date strings (inclusive) for a YYYY-MM string."""
    year, month = (int(x) for x in month_str.split("-"))
    first_day = date(year, month, 1)
    last_day_num = calendar.monthrange(year, month)[1]
    last_day = date(year, month, last_day_num)
    return first_day.isoformat(), last_day.isoformat()


def month_label(month_str: str) -> str:
    year, month = (int(x) for x in month_str.split("-"))
    return date(year, month, 1).strftime("%B %Y")


# --- Fetch daily digests for the month ---------------------------------------

def fetch_daily_digests_for_month(
    base_url: str,
    api_key: str,
    table: str,
    first_day: str,
    last_day: str,
) -> List[Dict[str, Any]]:
    url = f"{base_url}/rest/v1/{table}"
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
    }
    # PostgREST requires the same column filter passed as repeated query params
    # (e.g. ?date=gte.X&date=lte.Y) for an AND range condition; requests needs a
    # list of tuples (not a dict) to send duplicate keys.
    params = [
        ("select", "date,items,compiled"),
        ("date", f"gte.{first_day}"),
        ("date", f"lte.{last_day}"),
        ("order", "date.asc"),
    ]
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
    except requests.RequestException as e:
        print(f"[Supabase] Request error fetching daily digests: {e}")
        return []
    if r.status_code != 200:
        print(f"[Supabase] Failed to fetch daily digests: {r.status_code} - {r.text[:300]}")
        return []
    try:
        return r.json() or []
    except ValueError:
        print("[Supabase] Failed to decode daily digests JSON response.")
        return []


# --- Grouping & de-duplication ------------------------------------------------

def _normalize_topic_key(topic: str) -> str:
    return (topic or "general").strip().lower()


def _dedupe_key(item: Dict[str, Any]) -> str:
    url = (item.get("url") or "").strip().lower()
    if url:
        return url
    title = (item.get("title") or "").strip().lower()
    return title


def group_items_by_topic(daily_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group all daily-digest items by topic, deduplicating repeated stories.

    Returns: { topic_key: { "topic_title": str, "items": [ {title,url,published_at,summary,date}, ... ] } }
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in daily_rows:
        row_date = row.get("date")
        items = row.get("items") or []
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            summary = it.get("summary") or ""
            if "[ERROR:" in summary:
                continue
            topic_raw = it.get("topic") or "General"
            key = _normalize_topic_key(topic_raw)
            bucket = grouped.setdefault(key, {"topic_title": topic_raw, "items": [], "seen": set()})
            dedupe_key = _dedupe_key(it)
            if dedupe_key and dedupe_key in bucket["seen"]:
                continue
            if dedupe_key:
                bucket["seen"].add(dedupe_key)
            bucket["items"].append({
                "title": it.get("title"),
                "url": it.get("url"),
                "published_at": it.get("published_at"),
                "summary": summary,
                "date": row_date,
            })
    return grouped


# --- Summarization via Ollama -------------------------------------------------

def summarize_month_topic_with_ollama(model: str, topic: str, month_label_str: str, stories: List[Dict[str, Any]]) -> str:
    """Ask Ollama to synthesize a single month-in-review paragraph for one topic."""
    story_lines = []
    for s in stories:
        headline_or_title = s.get("title") or "Untitled"
        story_lines.append(f"- ({s.get('date', '')}) {headline_or_title}: {s.get('summary', '')}")
    stories_block = "\n".join(story_lines) if story_lines else "No stories recorded this month."

    prompt = f"""
You are a professional news editor creating a Monthly Summary digest for the category "{topic}".

You are given a list of daily news items from {month_label_str} for this single category only.
Several entries may refer to the same ongoing story on different days — treat those as one
storyline and describe its overall arc/outcome rather than repeating it multiple times.

Instructions:
1. Create a COMPLETE, engaging headline (8-15 words) capturing the month's single most important
   theme or development in this category.
2. Write ONE well-structured paragraph (5-8 sentences) that synthesizes the month's most important
   news in this category. Cover the most significant distinct stories, merge duplicate/related
   items into a single mention, and do NOT list the same event more than once.
3. Use third-person voice only. Stick to facts; no speculation or opinions.
4. Do NOT mention or reference other categories - stay strictly within "{topic}".
5. Do NOT repeat the headline text inside the summary paragraph.

Daily items for {topic} in {month_label_str}:
{stories_block[:6000]}

Required Format (provide exactly this):
Headline: [Your complete, engaging headline here]
Summary: [Your single synthesized paragraph here]
""".strip()

    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    endpoint = f"{host}/api/chat"
    payload = {
        "model": model or "gemma3:4b",
        "messages": [
            {"role": "system", "content": "You are a concise news editor writing a monthly review."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.1, "num_predict": 400},
        "stream": False,
    }

    for attempt in range(1, 4):
        print(f"[Ollama] Summarizing month for topic '{topic}' (attempt {attempt})...")
        try:
            resp = requests.post(endpoint, json=payload, timeout=120)
            if resp.status_code != 200:
                print(f"[Ollama] HTTP {resp.status_code} for topic {topic}: {resp.text[:200]}")
            else:
                try:
                    data = resp.json()
                except ValueError as e_json:
                    print(f"[Ollama] JSON decode error: {repr(e_json)}")
                    data = None
                if data:
                    message = data.get("message", {}).get("content") if isinstance(data, dict) else None
                    if message:
                        print(f"[Ollama] Success for topic: {topic}")
                        return message.strip()
                    print(f"[Ollama] API returned no message content for topic: {topic}")
        except requests.RequestException as e_http:
            print(f"[Ollama] HTTP request failed on attempt {attempt}: {repr(e_http)}")
        time.sleep(1)

    print(f"[Ollama] All attempts failed for topic: {topic}")
    return f"Headline: {topic} Monthly Summary\nSummary:\n• [ERROR: Could not summarize {topic} with {model or 'gemma3:4b'} after 3 attempts]"


# --- Supabase upsert -----------------------------------------------------------

def upsert_supabase_monthly_digest(
    base_url: str,
    service_role_key: str,
    table: str,
    month_str: str,
    compiled: str,
    items: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    if not base_url.startswith("http"):
        return False, "Invalid SUPABASE_URL; must start with http(s)."

    url = f"{base_url}/rest/v1/{table}"
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json",
        # Merge-duplicates requires a unique constraint on `month` in the target table.
        "Prefer": "return=representation,resolution=merge-duplicates",
    }
    payload = {
        "month": month_str,
        "compiled": compiled,
        "items": items,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code in (200, 201):
            return True, "Inserted monthly digest into Supabase."
        return False, f"Supabase insert failed: {r.status_code} - {r.text[:300]}"
    except requests.RequestException as e:
        return False, f"Supabase request error: {e}"


# --- Main ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Monthly Summary digest from daily digests.")
    parser.add_argument("--month", help="Target month as YYYY-MM. Defaults to the most recently completed month.")
    args = parser.parse_args()

    cfg = load_config()
    target_month = args.month or previous_month_str()
    try:
        first_day, last_day = month_bounds(target_month)
    except (ValueError, TypeError):
        print(f"[Error] Invalid --month value: {args.month!r}. Expected format YYYY-MM.")
        return 1
    label = month_label(target_month)
    print(f"[MonthlySummary] Target month: {target_month} ({label}), range {first_day}..{last_day}")

    ollama_model = cfg.get("OLLAMA_MODEL") or "gemma3:4b"
    ensure_ollama_ready(ollama_model, max_retries=30, timeout_secs=2)

    daily_table = cfg.get("SUPABASE_TABLE") or "daily_digests"
    monthly_table = cfg.get("SUPABASE_MONTHLY_TABLE") or MONTHLY_TABLE_DEFAULT

    missing = [k for k in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY") if not cfg.get(k)]
    if missing:
        print(f"Missing config values: {', '.join(missing)}. Check ven.env.")
        return 1

    rows = fetch_daily_digests_for_month(
        cfg["SUPABASE_URL"], cfg["SUPABASE_SERVICE_ROLE_KEY"], daily_table, first_day, last_day
    )
    if not rows:
        print(f"[MonthlySummary] No daily digests found for {target_month}; nothing to summarize.")
        return 0
    print(f"[MonthlySummary] Found {len(rows)} daily digest rows for {target_month}.")

    grouped = group_items_by_topic(rows)
    if not grouped:
        print(f"[MonthlySummary] No usable items found across {len(rows)} daily digests.")
        return 0

    # Preserve a stable, sensible topic ordering (matches the daily digest's category order).
    topic_order = ["general", "world", "us", "business", "technology", "entertainment", "sports", "science", "health"]
    ordered_keys = [k for k in topic_order if k in grouped] + [k for k in grouped.keys() if k not in topic_order]

    monthly_items: List[Dict[str, Any]] = []
    for key in ordered_keys:
        bucket = grouped[key]
        topic_title = "US" if key == "us" else bucket["topic_title"]
        stories = bucket["items"]
        summary = summarize_month_topic_with_ollama(ollama_model, topic_title, label, stories)
        if "[ERROR:" in summary:
            continue
        # Use the most recent story's URL as the representative link for this topic, if any.
        representative_url = next((s.get("url") for s in reversed(stories) if s.get("url")), None)
        monthly_items.append({
            "topic": topic_title,
            "title": f"{topic_title} — {label} Monthly Summary",
            "url": representative_url,
            "published_at": None,
            "summary": summary,
        })

    if not monthly_items:
        print("[MonthlySummary] All topic summarizations failed; aborting without publishing.")
        return 1

    compiled = compile_digest(monthly_items)

    ok, msg = upsert_supabase_monthly_digest(
        cfg["SUPABASE_URL"], cfg["SUPABASE_SERVICE_ROLE_KEY"], monthly_table, target_month, compiled, monthly_items
    )
    print(msg)
    if not ok:
        print("Note: Supabase insert failed. Ensure the monthly_digests table exists (see scripts/monthly_digests_schema.sql).")
        return 1

    # Generate and upload audio narration.
    try:
        narration = build_narration_text_from_compiled(
            compiled,
            intro=f"This is your Day2Day News Monthly Summary for {label}.",
            outro="That's everything for this month from Day2Day News.",
            empty_message="No monthly news is available.",
            debug_filename="monthly_narration_debug.txt",
        )
        audio_bytes = generate_audio_with_kokoro(narration, voice="am_michael")
        if audio_bytes:
            try:
                print(f"[Audio] Converting WAV ({len(audio_bytes)} bytes) to MP3...")
                wav_io = io.BytesIO(audio_bytes)
                audio_segment = AudioSegment.from_wav(wav_io)
                mp3_io = io.BytesIO()
                audio_segment.export(mp3_io, format="mp3", bitrate="128k")
                audio_bytes = mp3_io.getvalue()
                file_ext = "mp3"
                content_type = "audio/mpeg"
                print(f"[Audio] Converted to MP3 ({len(audio_bytes)} bytes)")
            except Exception as conv_err:
                print(f"[Audio] MP3 conversion failed ({conv_err}), using WAV fallback")
                file_ext = "wav"
                content_type = "audio/wav"

            bucket = AUDIO_BUCKET
            ensure_supabase_bucket(cfg["SUPABASE_URL"], cfg["SUPABASE_SERVICE_ROLE_KEY"], bucket)
            obj_path = f"monthly-{target_month}.{file_ext}"
            up_ok, up_msg = upload_audio_to_supabase(
                cfg["SUPABASE_URL"], cfg["SUPABASE_SERVICE_ROLE_KEY"], bucket, obj_path, audio_bytes, content_type
            )
            if not up_ok:
                print(up_msg)
                print("[Audio] Upload failed; waiting 30 seconds before one retry...")
                time.sleep(30)
                up_ok, up_msg = upload_audio_to_supabase(
                    cfg["SUPABASE_URL"], cfg["SUPABASE_SERVICE_ROLE_KEY"], bucket, obj_path, audio_bytes, content_type
                )
            print(up_msg)
            if up_ok:
                public_url = f"{cfg['SUPABASE_URL']}/storage/v1/object/public/{bucket}/{obj_path}"
                print(f"[Audio] Public URL: {public_url}")
        else:
            print("[Audio] Skipping upload because audio generation failed or returned empty data")
    except Exception as e:
        print(f"[Audio] Unexpected error during monthly audio pipeline: {e}")

    print(f"[MonthlySummary] Done for {target_month}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
