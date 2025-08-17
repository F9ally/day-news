import os
import json
import requests
from datetime import datetime

# Load environment variables from ven.env
env_path = os.path.join(os.path.dirname(__file__), '../ven.env')
with open(env_path) as f:
    for line in f:
        if '=' in line:
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

# Find the latest digest file in out/
out_dir = os.path.join(os.path.dirname(__file__), '../out')
digest_files = [f for f in os.listdir(out_dir) if f.endswith('.json')]
if not digest_files:
    print('No digest files found in out/.')
    exit(1)
latest_file = max(digest_files, key=lambda x: os.path.getmtime(os.path.join(out_dir, x)))
latest_path = os.path.join(out_dir, latest_file)

with open(latest_path, 'r', encoding='utf-8') as f:
    digest_data = json.load(f)

# Prepare payload for Supabase
data = {
    "date": digest_data.get("date", datetime.now().strftime('%Y-%m-%d')),
    "compiled": digest_data.get("compiled", ""),
    "items": json.dumps(digest_data.get("items", []))
}

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

url = f"{SUPABASE_URL}/rest/v1/daily_digests"
response = requests.post(url, headers=headers, json=data)

print(f"Status code: {response.status_code}")
print(f"Response: {response.text}")
