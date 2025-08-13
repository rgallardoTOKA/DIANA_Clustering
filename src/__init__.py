import os

from platformdirs import user_cache_dir

CACHE_DIR = user_cache_dir("DIANA_Cluster", "Rafael Gallardo")
os.makedirs(CACHE_DIR, exist_ok=True)
