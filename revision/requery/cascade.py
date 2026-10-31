"""
cascade.py — the manuscript's geocoding cascade, shared by the local-run scripts.
clean_address_variants() is byte-identical to scpark-network/pipeline/02_geocode_and_enrich.py.
geocode_with_step() additionally records WHICH variant succeeded (variant_idx) and the variant text.
Cache is a JSON file keyed by the query string; safe to interrupt (Ctrl+C) and resume.
"""
import json
import re
import time
from pathlib import Path

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "micro-geo-innovation-mapper/0.1 (dddsss5419@gmail.com)"}
MIN_INTERVAL = 1.1

INSTITUTION_KEYWORDS = re.compile(
    r'\b(university|institute|college|hospital|school|center|centre|'
    r'corp|inc|ltd|laboratory|lab|academy|foundation|clinic|company)\b',
    re.IGNORECASE,
)


def clean_address_variants(address):
    variants = [address]
    no_footnote = re.sub(r'^\d+(?=[A-Za-z])', '', address).strip()
    if no_footnote and no_footnote != address:
        variants.append(no_footnote)
    no_email = re.sub(r'Electronic address:\s*\S+', '', no_footnote, flags=re.IGNORECASE)
    no_email = re.sub(r'\S+@\S+', '', no_email).strip(' .;,')
    if no_email and no_email != no_footnote:
        variants.append(no_email)
    parts = [p.strip() for p in no_email.split(',') if p.strip()]
    inst_idx = None
    for i, p in enumerate(parts):
        if INSTITUTION_KEYWORDS.search(p):
            inst_idx = i
    if inst_idx is not None and inst_idx > 0:
        variants.append(', '.join(parts[inst_idx:]))
    for n in (3, 2):
        if len(parts) > n:
            variants.append(', '.join(parts[-n:]))
    seen, uniq = set(), []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


VARIANT_LABELS = ["raw", "footnote_stripped", "email_stripped", "from_last_institution_segment", "last_3_segments", "last_2_segments"]


def label_variant(address, idx):
    """Name the cascade step that produced variant idx (variants are de-duplicated, so map by content)."""
    v = clean_address_variants(address)
    target = v[idx]
    no_footnote = re.sub(r'^\d+(?=[A-Za-z])', '', address).strip()
    no_email = re.sub(r'Electronic address:\s*\S+', '', no_footnote, flags=re.IGNORECASE)
    no_email = re.sub(r'\S+@\S+', '', no_email).strip(' .;,')
    parts = [p.strip() for p in no_email.split(',') if p.strip()]
    if idx == 0:
        return "raw"
    if target == no_footnote:
        return "footnote_stripped"
    if target == no_email:
        return "email_stripped"
    if len(parts) > 3 and target == ', '.join(parts[-3:]):
        return "last_3_segments"
    if len(parts) > 2 and target == ', '.join(parts[-2:]):
        return "last_2_segments"
    return "from_last_institution_segment"


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.d = json.load(open(self.path, encoding="utf-8")) if self.path.exists() else {}

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        json.dump(self.d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
        tmp.replace(self.path)          # atomic: an interrupted save never corrupts the cache


def nominatim(q, cache, max_retries=3):
    """Single query, cached. Returns [lat, lon, display_name, country_code] or None."""
    if q in cache.d:
        return cache.d[q]
    params = {"q": q, "format": "jsonv2", "limit": 1, "addressdetails": 1}
    for attempt in range(max_retries):
        try:
            r = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            res = r.json()
            time.sleep(MIN_INTERVAL)
            val = None
            if res:
                a = res[0]
                val = [float(a["lat"]), float(a["lon"]), a.get("display_name", ""),
                       (a.get("address") or {}).get("country_code", "").upper()]
            cache.d[q] = val
            cache.save()
            return val
        except Exception as e:
            print(f"  [retry {attempt + 1}] {q[:60]}: {e}")
            time.sleep(2 ** attempt)
    return "NETWORK_FAIL"


def geocode_with_step(address, cache):
    """Run the cascade; return dict(success, variant_idx, step, lat, lon, cc, variant_text)."""
    for vi, v in enumerate(clean_address_variants(address)):
        res = nominatim(v, cache)
        if res == "NETWORK_FAIL":
            return {"success": None, "variant_idx": None, "step": "network_fail"}
        if res:
            return {"success": True, "variant_idx": vi, "step": label_variant(address, vi),
                    "lat": res[0], "lon": res[1], "cc": res[3], "variant_text": v}
    return {"success": False, "variant_idx": None, "step": "all_failed"}
