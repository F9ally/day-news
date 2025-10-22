import re
import os

# Load credentials from ven.env
venv_path = os.path.join(os.path.dirname(__file__), '../ven.env')
creds = {}
with open(venv_path, encoding='utf-8') as f:
    for line in f:
        if '=' in line:
            k, v = line.strip().split('=', 1)
            creds[k.strip()] = v.strip()

supabase_url = creds.get('SUPABASE_URL', '')
supabase_key = creds.get('SUPABASE_ANON_KEY', '')

# Replace placeholders in index.html
index_path = os.path.join(os.path.dirname(__file__), '../index.html')
with open(index_path, encoding='utf-8') as f:
    html = f.read()

html = re.sub(r'window\.SUPABASE_URL\s*=\s*"\{\{SUPABASE_URL\}\}";', f'window.SUPABASE_URL = "{supabase_url}";', html)
html = re.sub(r'window\.SUPABASE_KEY\s*=\s*"\{\{SUPABASE_ANON_KEY\}\}";', f'window.SUPABASE_KEY = "{supabase_key}";', html)

with open(index_path, 'w', encoding='utf-8') as f:
    f.write(html)

print('Supabase credentials injected into index.html.')
