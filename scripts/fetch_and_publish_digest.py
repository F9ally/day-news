import time
start = time.time()
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


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
        os.environ.setdefault(k, v)

    cfg = {
        "NEWS_API_URL": (env("NEWS_API_URL", "") or "").rstrip("/"),
        "NEWS_API_KEY": env("NEWS_API_KEY", ""),
        "SUPABASE_URL": (env("SUPABASE_URL", "") or "").rstrip("/"),
        "SUPABASE_ANON_KEY": env("SUPABASE_ANON_KEY", ""),
        "OLLAMA_MODEL": env("OLLAMA_MODEL", "mistral"),
        "SUPABASE_TABLE": env("SUPABASE_TABLE", "daily_digests"),
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


# --- News API fetch ---------------------------------------------------------

def try_news_api(
    base_url: str,
    api_key: str,
    query: Optional[str] = None,
    category: Optional[str] = None,
    language: str = "en",
) -> Optional[Dict[str, Any]]:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "day-news/1.0 (+https://github.com/F9ally/day-news)",
    })

    # Use only /search endpoint with category and api_key
    base = base_url.rstrip("/")
    url = f"{base}/articles"
    params = {
        "category": category,
        "limit": 1,
        "offset": 0,
        "language": language,
        "api_key": api_key,
    }
    print(f"Requesting: {url} with params: {params}")
    try:
        r = session.get(url, params=params, timeout=10)
        print(f"Response status: {r.status_code}")
        if r.status_code != 200:
            print(f"Error response: {r.text}")
            return None
        data = r.json()
        print(f"Response data: {json.dumps(data)[:500]}")

        article = None
        if isinstance(data, dict) and isinstance(data.get("articles"), list) and data["articles"]:
            article = data["articles"][0]

        if not article:
            print("No article found in response.")
            return None

        title = article.get("title") or "Untitled"
        description = article.get("description") or ""
        url_field = article.get("link")
        published_at = article.get("published_date")
        content = article.get("content") or description

        return {
            "title": str(title) if title else "Untitled",
            "description": str(description) if description else "",
            "url": str(url_field) if url_field else None,
            "published_at": str(published_at) if published_at else None,
            "content": str(content) if content else "",
        }
    except requests.RequestException as e:
        print(f"RequestException: {e}")
        return None
    except ValueError as e:
        print(f"ValueError: {e}")
        return None

def try_news_api_with_retries(base_url, api_key, category, language="en", max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        print(f"[{category}] Attempt {attempt} of {max_attempts}")
        result = try_news_api(base_url, api_key, query=None, category=category, language=language)
        if result:
            return result
        print(f"[{category}] No article found, retrying...")
    print(f"[{category}] All attempts failed.")
    return None
    """
    Fetch a single article for the given query using a few common patterns.
    Returns a normalized article dict or None.

    Normalized keys: title, description, url, published_at, content
    """
    if not base_url:
        return None

    session = requests.Session()
    session.headers.update({
        "User-Agent": "day-news/1.0 (+https://github.com/F9ally/day-news)",
    })

    # Use only /search endpoint with category and api_key
    base = base_url.rstrip("/")
    url = f"{base}/search"
    params = {
        "category": category,
        "api_key": api_key,
        "page_size": 1,
        "language": language,
    }
    print(f"Requesting: {url} with params: {params}")
    try:
        r = session.get(url, params=params, timeout=10)
        print(f"Response status: {r.status_code}")
        if r.status_code != 200:
            print(f"Error response: {r.text}")
            return None
        data = r.json()
        print(f"Response data: {json.dumps(data)[:500]}")

        article = None
        if isinstance(data, list) and data:
            article = data[0]
        elif isinstance(data, dict):
            if isinstance(data.get("articles"), list) and data["articles"]:
                article = data["articles"][0]
            elif isinstance(data.get("news"), list) and data["news"]:
                article = data["news"][0]
            elif isinstance(data.get("result"), list) and data["result"]:
                article = data["result"][0]
            elif isinstance(data.get("results"), list) and data["results"]:
                article = data["results"][0]
            elif isinstance(data.get("data"), list) and data["data"]:
                article = data["data"][0]

        if not article:
            print("No article found in response.")
            return None

        title = article.get("title") or article.get("name") or "Untitled"
        description = article.get("description") or article.get("summary") or ""
        url_field = (
            article.get("url")
            or article.get("link")
            or article.get("source_url")
        )
        published_at = (
            article.get("publishedAt")
            or article.get("published_at")
            or article.get("date")
            or article.get("pubDate")
        )
        content = (
            article.get("content")
            or article.get("full_content")
            or article.get("text")
            or description
        )

        return {
            "title": str(title) if title else "Untitled",
            "description": str(description) if description else "",
            "url": str(url_field) if url_field else None,
            "published_at": str(published_at) if published_at else None,
            "content": str(content) if content else "",
        }
    except requests.RequestException as e:
        print(f"RequestException: {e}")
        return None
    except ValueError as e:
        print(f"ValueError: {e}")
        return None


# --- Summarization via Ollama ----------------------------------------------

def summarize_with_ollama(model: str, title: str, content: str, url: Optional[str]) -> str:
    """
    Use local Ollama (Mistral) to summarize a single article.
    Falls back to a simple heuristic summary if ollama isn't reachable.
    """
    prompt = f"""
    Summarize the following news article clearly and concisely in 3-5 bullet points.
    Include 1 short headline style title at the top.
    Keep it factual, neutral, and under 120 words total.

    Title: {title}
    URL: {url or 'N/A'}

    Article content:
    {content}
    """.strip()

    for attempt in range(1, 4):
        try:
            print(f"[Ollama] Summarizing (attempt {attempt}) for: {title}")
            import ollama  # type: ignore
            resp = ollama.chat(
                model="mistral:7b",
                messages=[
                    {"role": "system", "content": "You are a concise news editor."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.2},
            )
            out = resp.get("message", {}).get("content")
            if out:
                print(f"[Ollama] Success for: {title}")
                return out.strip()
        except Exception as e:
            print(f"[Ollama] Error (attempt {attempt}) for {title}: {e}")
    print(f"[Ollama] All attempts failed for: {title}")
    return f"{title}\n- [ERROR: Could not summarize with Mistral 7B after 3 attempts]"


# --- Formatting -------------------------------------------------------------

def compile_digest(summaries: List[Tuple[str, str]]) -> str:
    """
    Given a list of (topic_title, summary_text), compile into a single numbered digest
    with each entry as a paragraph, titled and numbered.
    """
    paras = []
    for i, (topic_title, summary) in enumerate(summaries, start=1):
        paras.append(f"{i}. {topic_title}\n{summary.strip()}")
    return "\n\n".join(paras)


# --- Supabase insert --------------------------------------------------------

def upsert_supabase_digest(
    base_url: str,
    anon_key: str,
    table: str,
    compiled: str,
    items: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    if not base_url.startswith("http"):
        return False, "Invalid SUPABASE_URL; must start with http(s)."

    url = f"{base_url}/rest/v1/{table}?return=representation"
    headers = {
        "apikey": anon_key,
        "Authorization": f"Bearer {anon_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    payload = {
        "date": datetime.now(timezone.utc).date().isoformat(),
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
    date_str = datetime.now().strftime("%Y-%m-%d")
    txt_path = os.path.join(out_dir, f"digest_{date_str}.txt")
    json_path = os.path.join(out_dir, f"digest_{date_str}.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(compiled)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return txt_path


# --- Main -------------------------------------------------------------------

def main() -> int:
    cfg = load_config()
    missing = [k for k in ("NEWS_API_URL", "NEWS_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY") if not cfg.get(k)]
    if missing:
        print(f"Missing required config values: {', '.join(missing)}. Check ven.env.")
        # continue anyway; user may want to run partially

    topics = get_topics()

    fetched: List[Dict[str, Any]] = []
    failed_categories: List[str] = []
    for t in topics:
        cat = t.get("category")
        article = try_news_api_with_retries(cfg["NEWS_API_URL"], cfg["NEWS_API_KEY"], category=cat, language="en", max_attempts=3)
        if not article:
            failed_categories.append(cat)
            article = {
                "title": f"No article found for category {cat}",
                "description": "",
                "url": None,
                "published_at": None,
                "content": "",
            }
        fetched.append({
            "topic": cat,
            **article,
        })

    # Email notification if any failures
    if failed_categories:
        send_failure_email(failed_categories)

    # Summarize each
    summaries: List[Tuple[str, str]] = []
    items_for_supabase: List[Dict[str, Any]] = []
    for item in fetched:
        topic_title = item["topic"].replace("-", "/").title()
        summary = summarize_with_ollama(cfg["OLLAMA_MODEL"], item.get("title", "Untitled"), item.get("content", ""), item.get("url"))
        summaries.append((topic_title, summary))
        items_for_supabase.append({
            "topic": topic_title,
            "title": item.get("title"),
            "url": item.get("url"),
            "published_at": item.get("published_at"),
            "summary": summary,
        })

    compiled = compile_digest(summaries)

    # Insert into Supabase
    ok, msg = upsert_supabase_digest(
        cfg["SUPABASE_URL"], cfg["SUPABASE_ANON_KEY"], cfg["SUPABASE_TABLE"], compiled, items_for_supabase
    )
    print(msg)

    # Always save local fallback
    out_path = save_local_fallback(compiled, items_for_supabase)
    print(f"Saved local digest to: {out_path}")

    if not ok:
        print("Note: Supabase insert failed. Ensure your SUPABASE_URL is like https://<ref>.supabase.co and the anon key is valid, and that the table exists.")
        print("Expected table schema (SQL):\n"
              "create table if not exists public.daily_digests (\n"
              "  id uuid primary key default gen_random_uuid(),\n"
              "  created_at timestamptz default now(),\n"
              "  date date,\n"
              "  compiled text,\n"
              "  items jsonb\n"
              ");")

    print(f"Total runtime: {time.time() - start:.2f} seconds")
    return 0


def send_failure_email(failed_categories):
    sender = "justwebsites.contact@gmail.com"
    recipient = "justwebsites.contact@gmail.com"
    subject = "Day News Digest: API Failure Notification"
    body = f"The following categories failed to fetch news after 3 attempts:\n\n" + ", ".join(failed_categories)
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = formataddr(("Day News", sender))
    msg["To"] = recipient

    try:
        # For Gmail, you may need an app password and enable less secure apps
        smtp = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        smtp.login(sender, "YOUR_APP_PASSWORD_HERE")
        smtp.sendmail(sender, [recipient], msg.as_string())
        smtp.quit()
        print(f"Failure notification sent to {recipient}")
    except Exception as e:
        print(f"Failed to send email notification: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
