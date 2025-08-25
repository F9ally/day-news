import time
start = time.time()
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import re
import html as html_lib


# --- Config helpers ---------------------------------------------------------

def load_env_file(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip()
    return values


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


def load_config() -> Dict[str, str]:
    # Load from ven.env first, then overlay with actual environment variables
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root, "ven.env")
    file_vals = load_env_file(env_path)
    for k, v in file_vals.items():
        # If the environment does not have this var or it's empty, populate from ven.env
        if not os.environ.get(k):
            os.environ[k] = v
    # Debug: show which keys we parsed from ven.env
    try:
        parsed = ", ".join(sorted(file_vals.keys()))
        print(f"[Config] Parsed from ven.env: {parsed}")
    except Exception:
        pass

    # Build cfg preferring env vars; fallback to ven.env values if env is empty
    def pick(name: str, default: str = "") -> str:
        v = os.environ.get(name)
        if v:
            return v
        return file_vals.get(name, default)

    cfg = {
        # Default to World News API if not provided
        "NEWS_API_URL": (pick("NEWS_API_URL", "https://api.worldnewsapi.com").rstrip("/")),
        "NEWS_API_KEY": pick("NEWS_API_KEY", ""),
        "SUPABASE_URL": (pick("SUPABASE_URL", "").rstrip("/")),
        "SUPABASE_SERVICE_ROLE_KEY": pick("SUPABASE_SERVICE_ROLE_KEY", ""),
        "SUPABASE_ANON_KEY": pick("SUPABASE_ANON_KEY", ""),
        "OLLAMA_MODEL": pick("OLLAMA_MODEL", "gemma:2b"),
        "SUPABASE_TABLE": pick("SUPABASE_TABLE", "daily_digests"),
    }
    return cfg


# --- Domain -----------------------------------------------------------------

Topic = Dict[str, str]


def get_topics() -> List[Topic]:
    # 10 requested categories (use category param where supported by the API)
    # categories: technology, business, politics, health, science, entertainment, sports, environment, education, world
    return [
        {"category": "world"},
        {"category": "politics"},
        {"category": "business"},
        {"category": "technology"},
        {"category": "entertainment"},
        {"category": "sports"},
        {"category": "health"},
        {"category": "environment"},
        {"category": "science"},
        {"category": "education"},
    ]

# Simple keyword heuristics to avoid off-topic picks for certain categories
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "environment": [
        "climate", "emission", "wildlife", "ecosystem", "renewable", "conservation",
        "biodiversity", "pollution", "carbon", "deforestation", "sustainability", "habitat"
    ],
    "education": [
        "school", "student", "university", "college", "curriculum", "teacher",
        "education", "exam", "classroom", "tuition", "degree", "campus"
    ],
    "entertainment": [
        "film", "movie", "tv", "series", "celebrity", "music", "concert", "festival",
        "box office", "trailer", "actor", "actress", "netflix", "hbo"
    ],
}


def _is_article_relevant(category: Optional[str], article: Dict[str, Any]) -> bool:
    if not category:
        return True
    cat = category.lower()
    keywords = CATEGORY_KEYWORDS.get(cat)
    if not keywords:
        return True  # only enforce for selected categories
    text = " ".join([
        str(article.get("title") or ""),
        str(article.get("description") or ""),
        str(article.get("content") or ""),
    ]).lower()
    hits = sum(1 for kw in keywords if kw in text)
    return hits >= 1



# NewsData.io API endpoint and key
NEWSDATA_API_URL = "https://newsdata.io/api/1/latest"
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

def try_newsdata_api(category, page=1):
    params = {
        "apikey": NEWSDATA_API_KEY,
        "language": "en",
        "category": category,
        "size": 10,  # free user default
        "page": page,
    }
    response = requests.get(NEWSDATA_API_URL, params=params)
    quota_left = response.headers.get("X-RateLimit-Remaining")
    print(f"Quota left: {quota_left}")
    if response.status_code == 200:
        data = response.json()
        articles = data.get("results", [])
        if articles:
            return articles[0]  # return first article
        else:
            print(f"No articles found for category {category}")
            return None
    else:
        print(f"Error: {response.status_code} {response.text}")
        return None

def try_newsdata_api_with_retries(category, max_attempts=3):
    seen_urls = set()
    for attempt in range(max_attempts):
        article = try_newsdata_api(category, page=attempt+1)
        if article and article.get("link") not in seen_urls:
            seen_urls.add(article.get("link"))
            return article
        print(f"[{category}] No article found, retrying...")
    print(f"[{category}] All attempts failed.")
    return None


# --- Summarization via Ollama ----------------------------------------------

def summarize_with_ollama(model: str, topic: str, title: str, content: str, url: Optional[str]) -> str:
    """
    Use local Ollama (Mistral) to summarize a single article.
    Falls back to a simple heuristic summary if ollama isn't reachable.
    """
    prompt = f"""
You are a precise news editor. Summarize the article strictly for the category: {topic}.

Hard rules:
- Use third-person only (no first-person like "I", "we").
- Keep focus on the given category; do not include other category labels (no "Politics:", "Economy:").
- Be factual and concise; avoid speculation.
- No questions to the reader; no calls to action.

Return ONLY in this format (no extra text):
Headline: <a concise headline for this article, <= 120 chars>
Summary:
• point 1 (a short, factual sentence)
• point 2 (a short, factual sentence)
• point 3 (optional; include only if meaningful)

Title: {title}
URL: {url or 'N/A'}
Article content (truncated):
{(content[:4000] + '...') if content and len(content) > 4000 else (content or '')}
""".strip()

    for attempt in range(1, 4):
        print(f"[Ollama] Summarizing (attempt {attempt}) for: {title}")
        import subprocess
        try:
            # Use the CLI: ollama run gemma:2b "<prompt>"
            cmd = ["ollama", "run", model or "gemma:2b", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=60)
            # If encoding fails, try decoding manually
            output = result.stdout
            if not output:
                try:
                    output = result.stdout.decode("utf-8", errors="replace") if hasattr(result.stdout, 'decode') else str(result.stdout)
                except Exception as e_dec:
                    print(f"[Ollama] CLI decode error: {repr(e_dec)}")
                    output = str(result.stdout)
            if result.returncode == 0 and output.strip():
                print(f"[Ollama] Success via CLI for: {title}")
                return output.strip()
            else:
                err = result.stderr
                if not err:
                    try:
                        err = result.stderr.decode("utf-8", errors="replace") if hasattr(result.stderr, 'decode') else str(result.stderr)
                    except Exception as e_dec:
                        print(f"[Ollama] CLI stderr decode error: {repr(e_dec)}")
                        err = str(result.stderr)
                print(f"[Ollama] CLI failed: {err.strip()}")
        except Exception as e_cli:
            print(f"[Ollama] CLI call failed with error: {repr(e_cli)}")

        # Fallback to Python client chat() if CLI fails
        try:
            import ollama  # type: ignore
            resp = ollama.chat(
                model=model or "gemma:2b",
                messages=[
                    {"role": "system", "content": "You are a concise news editor."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.1, "num_predict": 256},
            )
            out = resp.get("message", {}).get("content") if isinstance(resp, dict) else None
            if out:
                print(f"[Ollama] Success via chat() for: {title}")
                return out.strip()
            else:
                print(f"[Ollama] chat() returned no content for: {title}")
        except Exception as e_chat:
            print(f"[Ollama] chat() failed with error: {repr(e_chat)}")
        print(f"[Ollama] Tip: Ensure the Ollama server is running (e.g., start the daemon) and the model is available: '{model or 'gemma:2b'}'.")
    print(f"[Ollama] All attempts failed for: {title}")
    return f"Headline: {title}\nSummary:\n• [ERROR: Could not summarize with Gemma 2B after 3 attempts]"


# --- Formatting -------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove common markdown syntax and collapse whitespace."""
    if not text:
        return ""
    t = text
    # Remove fenced code blocks
    t = re.sub(r"```[\s\S]*?```", " ", t)
    # Strip headings like #, ## at start of lines
    t = re.sub(r"^\s*#{1,6}\s+", "", t, flags=re.MULTILINE)
    # Bold/italic markers
    t = t.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    # Inline code backticks
    t = t.replace("`", "")
    # Replace markdown bullets with separators
    t = re.sub(r"^[\s>*\-•]+", "• ", t, flags=re.MULTILINE)
    # Remove stray markdown links [text](url) -> text
    t = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", t)
    # Collapse multiple newlines to a single space
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _clean_for_html(summary: str) -> str:
    """Sanitize model output for safe HTML embedding: strip markdown and escape HTML."""
    # First, strip markdown and decode any HTML entities the model may have emitted (e.g., &#x27;)
    clean = _strip_markdown(summary)
    clean = html_lib.unescape(clean)
    # Escape any HTML to avoid injecting tags
    clean = html_lib.escape(clean)
    return clean


def _is_recent(published_at: Optional[str], hours: int = 30) -> bool:
    if not published_at:
        return False
    try:
        # Attempt to parse common ISO formats
        s = published_at.strip()
        # Normalize 'Z' suffix
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - dt.astimezone(timezone.utc)
        return age.total_seconds() <= hours * 3600
    except Exception:
        return False


def _is_digest_day_utc(published_at: Optional[str]) -> bool:
    if not published_at:
        return False
    try:
        s = published_at.strip()
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        # For NewsData.io, just check if published within last 48 hours
        age = datetime.now(timezone.utc) - dt_utc
        return age.total_seconds() <= 48 * 3600
    except Exception:
        return False


def _extract_headline_and_points(cleaned_text: str) -> Tuple[str, List[str]]:
    """Parse the model output to extract the Headline and bullet points."""
    headline = ""
    points: List[str] = []
    # Find Headline: line
    m = re.search(r"Headline:\s*(.+)", cleaned_text, re.IGNORECASE)
    if m:
        headline = m.group(1).strip()
    # Find Summary section and bullets
    # Accept both '•' bullets and lines starting with hyphen/asterisk
    summary_part = cleaned_text
    sm = re.search(r"Summary:\s*(.*)$", cleaned_text, re.IGNORECASE | re.DOTALL)
    if sm:
        summary_part = sm.group(1).strip()
    for line in summary_part.split("\n"):
        ln = line.strip()
        if not ln:
            continue
        # Normalize bullet markers
        ln = re.sub(r"^(?:[\-\*\u2022]|•)\s*", "", ln)
        points.append(ln)
    # If no bullets found but text exists, split by '•'
    if not points and "•" in summary_part:
        points = [s.strip() for s in summary_part.split("•") if s.strip()]
    return headline, points


def compile_digest(items: List[Dict[str, str]]) -> str:
    """
    Given a list of items with keys: topic (title-cased), title, summary.
    Build safe HTML with topic header, bold article headline, and per-line bullet paragraphs.
    """
    topic_emojis = {
        "world": "🌍",
        "politics": "📰",
        "business": "💰",
        "technology": "💻",
        "entertainment": "🎭",
        "sports": "🏅",
        "health": "🩺",
        "environment": "🌱",
        "science": "🔬",
        "education": "🎓",
    }
    seen = set()
    html_blocks = []
    # Helper to strip leading topic labels the model may add
    topic_label_re = re.compile(r"^(World|Politics|Business|Technology|Entertainment|Sports|Health|Environment|Science|Education)\s*:\s*", re.IGNORECASE)
    for it in items:
        topic_title = it.get("topic", "General")
        summary = it.get("summary", "")
        art_title = it.get("title", "Untitled")
        key = topic_title.lower()
        if key in seen:
            continue
        seen.add(key)
        emoji = topic_emojis.get(key, "❓")
        # Skip error summaries entirely from the website output
        if "[ERROR:" in summary:
            continue
        # Remove a leading topic label if present
        no_label = topic_label_re.sub("", summary.strip())
        safe = _clean_for_html(no_label)
        # Extract headline + bullet points from the cleaned text (already HTML-escaped)
        headline, points = _extract_headline_and_points(safe)
        # Fallback to original article title as headline if parsing failed or headline empty
        if not headline:
            headline = html_lib.escape(art_title or "Untitled")
        # Remove points that duplicate the headline
        def _norm(x: str) -> str:
            return re.sub(r"[\s\.;:\-]+$", "", x.strip().lower())
        norm_head = _norm(headline)
        seen_pts = set()
        deduped: List[str] = []
        for p in points:
            np = _norm(p)
            if not np or np == norm_head or np in seen_pts:
                continue
            seen_pts.add(np)
            deduped.append(p)
        points = deduped
        # Build paragraphs for points
        point_ps = "".join(f"<p>{p}</p>" for p in points if p)
        safe_url = None
        if it.get("url"):
            try:
                safe_url = html_lib.escape(str(it.get("url")), quote=True)
            except Exception:
                safe_url = None
        # Headings with optional links
        if safe_url:
            h2_html = f'<h2><a href="{safe_url}" target="_blank" rel="noopener noreferrer">{emoji} {html_lib.escape(topic_title)}</a></h2>'
            h3_html = f'<h3 class="digest-headline"><a href="{safe_url}" target="_blank" rel="noopener noreferrer">{headline}</a></h3>'
        else:
            h2_html = f'<h2>{emoji} {html_lib.escape(topic_title)}</h2>'
            h3_html = f'<h3 class="digest-headline">{headline}</h3>'
        block = (
            f'<section class="digest-topic">'
            f'{h2_html}'
            f'{h3_html}'
            f'<div class="digest-points">{point_ps}</div>'
            f'</section>'
        )
        html_blocks.append(block)
    return "\n".join(html_blocks)
# --- Supabase insert --------------------------------------------------------

def upsert_supabase_digest(
    base_url: str,
    service_role_key: str,
    table: str,
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
        # Ask PostgREST to return the inserted row and merge duplicates by a unique constraint if present
        "Prefer": "return=representation,resolution=merge-duplicates",
    }

    # Store canonical ISO date for reliable filtering on the website
    formatted_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload = {
        "date": formatted_date,
        "compiled": compiled,
        "items": items,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code in (200, 201):
            return True, "Inserted digest into Supabase."
        else:
            return False, f"Supabase insert failed: {r.status_code} - {r.text[:300]}"
    except requests.RequestException as e:
        return False, f"Supabase request error: {e}"


# --- Local fallback output --------------------------------------------------

def save_local_fallback(compiled: str, items: List[Dict[str, Any]]) -> str:
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    os.makedirs(out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%d %B %Y")
    html_path = os.path.join(out_dir, f"digest_{date_str}.html")
    json_path = os.path.join(out_dir, f"digest_{date_str}.json")
    # Wrap the digest in a basic HTML template
    html_template = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Day News Digest - {date_str}</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #fafafa; color: #222; margin: 2em; }}
        .digest-topic {{ margin-bottom: 2em; padding: 1em; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #eee; }}
        h2 {{ margin-top: 0; font-size: 1.3em; }}
        .digest-error {{ color: #b00; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Day News Digest <span style=\"font-size:0.7em;color:#888\">{date_str}</span></h1>
    {compiled}
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return html_path


# --- Main -------------------------------------------------------------------

def main() -> int:
    cfg = load_config()
    def _mask(val: Optional[str]) -> str:
        if not val:
            return "<empty>"
        v = str(val)
        return (v[:4] + "..." + v[-4:]) if len(v) > 8 else "<short>"
    print(f"[Config] SUPABASE_URL={cfg.get('SUPABASE_URL','')} TABLE={cfg.get('SUPABASE_TABLE','')} KEY={_mask(cfg.get('SUPABASE_SERVICE_ROLE_KEY'))}")
    missing = [k for k in ("NEWS_API_URL", "NEWS_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY") if not cfg.get(k)]
    if missing:
        print(f"Missing required config values: {', '.join(missing)}. Check ven.env.")
        # continue anyway; user may want to run partially

    topics = get_topics()

    fetched: List[Dict[str, Any]] = []
    failed_categories: List[str] = []
    seen_urls: set = set()
    last_quota_seen: Optional[str] = None
    for t in topics:
        cat = t.get("category")
        article = try_newsdata_api_with_retries(cat, max_attempts=3)
        if not article:
            failed_categories.append(cat)
            article = {
                "title": f"No article found for category {cat}",
                "description": "",
                "url": None,
                "published_at": None,
                "content": "",
            }
        else:
            if article.get("url"):
                seen_urls.add(article["url"])
            if article.get("quota_left") is not None:
                last_quota_seen = str(article["quota_left"])
        fetched.append({
            "topic": cat,
            **article,
        })

    # Email notification if any failures
    if failed_categories:
        send_failure_email(failed_categories)

    # Summarize each
    items_for_supabase: List[Dict[str, Any]] = []
    for item in fetched:
        topic_title = item["topic"].replace("-", "/").title()
        summary = summarize_with_ollama(
            cfg.get("OLLAMA_MODEL", "gemma:2b"),
            topic_title,
            item.get("title", "Untitled"),
            item.get("content", ""),
            item.get("url"),
        )
        # Only include successful summaries in persisted items
        if "[ERROR:" not in summary:
            items_for_supabase.append({
                "topic": topic_title,
                "title": item.get("title"),
                "url": item.get("url"),
                "published_at": item.get("published_at"),
                "summary": summary,
            })

    compiled = compile_digest(items_for_supabase)

    # Insert into Supabase even if some summaries failed; we exclude failures from items/compiled
    ok, msg = upsert_supabase_digest(
        cfg["SUPABASE_URL"], cfg["SUPABASE_SERVICE_ROLE_KEY"], cfg["SUPABASE_TABLE"], compiled, items_for_supabase
    )
    print(msg)
    if not ok:
        print("Note: Supabase insert failed. Ensure your SUPABASE_URL is like https://<ref>.supabase.co and the key/table are valid.")

    # Always save local fallback
    out_path = save_local_fallback(compiled, items_for_supabase)
    print(f"Saved local digest to: {out_path}")

    # Show remaining World News API daily quota if observed
    if last_quota_seen is not None:
        print(f"[API] X-API-Quota-Left today: {last_quota_seen}")

    print(f"Total runtime: {time.time() - start:.2f} seconds")
    return 0


def send_failure_email(failed_categories):
    subject = "Day News Digest: API Failure Notification"
    body = (
        "The following categories failed to fetch news after 3 attempts:\n\n" + ", ".join(failed_categories)
    )

    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "465"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    sender = os.environ.get("SMTP_SENDER")
    recipient = os.environ.get("SMTP_TO")

    if not all([smtp_host, smtp_user, smtp_pass, sender, recipient]):
        print("Email notification skipped: SMTP credentials are not configured.")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = formataddr(("Day News", sender))
    msg["To"] = recipient

    try:
        smtp = smtplib.SMTP_SSL(smtp_host, smtp_port)
        smtp.login(smtp_user, smtp_pass)
        smtp.sendmail(sender, [recipient], msg.as_string())
        smtp.quit()
        print(f"Failure notification sent to {recipient}")
    except Exception as e:
        print(f"Failed to send email notification: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
