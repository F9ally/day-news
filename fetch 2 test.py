import os
import requests
from datetime import datetime, timedelta
from supabase import create_client, Client

# ----------------------------
# Configuration
# ----------------------------
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-service-role-key"
NEWS_API_KEY = "your-news-api-key"
NEWS_QUERY = "technology OR AI"
ARTICLES_TO_FETCH = 10

# Paths for storing already seen URLs
SEEN_URLS_FILE = "seen_urls.txt"

# ----------------------------
# Supabase client
# ----------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------
# Helper functions
# ----------------------------
def load_seen_urls():
    if os.path.exists(SEEN_URLS_FILE):
        with open(SEEN_URLS_FILE, "r") as f:
            return set(line.strip() for line in f.readlines())
    return set()

def save_seen_urls(urls):
    with open(SEEN_URLS_FILE, "w") as f:
        for url in urls:
            f.write(url + "\n")

def fetch_recent_articles():
    # Use today’s date for filtering
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={NEWS_QUERY}&"
        f"from={yesterday.isoformat()}&"
        f"to={today.isoformat()}&"
        f"pageSize={ARTICLES_TO_FETCH}&"
        f"sortBy=publishedAt&"
        f"apiKey={NEWS_API_KEY}"
    )
    
    response = requests.get(url)
    data = response.json()
    
    if data.get("status") != "ok":
        print("Error fetching articles:", data)
        return []
    
    return data.get("articles", [])

def summarize_article(article_text):
    # Placeholder: Replace with your local Gemma 2B summarization logic
    # For example, call your local model here
    return f"Summary of: {article_text[:100]}..."  # Short mock summary

def main():
    seen_urls = load_seen_urls()
    articles = fetch_recent_articles()
    
    new_urls = set()
    for article in articles:
        url = article.get("url")
        if not url or url in seen_urls:
            continue

        title = article.get("title", "")
        content = article.get("content", "") or article.get("description", "")
        published_at = article.get("publishedAt", datetime.utcnow().isoformat())
        
        summary = summarize_article(content)
        
        # Insert into Supabase
        supabase.table("news_summaries").insert({
            "title": title,
            "url": url,
            "summary": summary,
            "published_at": published_at
        }).execute()
        
        seen_urls.add(url)
        new_urls.add(url)
    
    # Update local seen URLs file
    if new_urls:
        save_seen_urls(seen_urls)
    print(f"Processed {len(new_urls)} new articles.")

if __name__ == "__main__":
    main()
