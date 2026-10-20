# -*- coding: utf-8 -*-
"""
Logika scrapera OKO.press (https://oko.press/najnowsze) - wersja modulowa.

W przeciwienstwie do gazet (scrape_newspapers) OKO.press to aplikacja Next.js,
ktora osadza gotowe dane w bloku <script id="__NEXT_DATA__"> (cache Apollo/GraphQL).
Nie parsujemy HTML-a DOM, tylko ten JSON. Kazda strona listy = 18 artykulow,
paginacja ?page=N, posortowane malejaco po dacie publikacji.

Zapis przyrostowy do CSV (utf-8-sig): funkcja wznawia sie po URL-ach juz zapisanych,
a przy podanym `since` przerywa, gdy trafi na artykul starszy niz data graniczna.
Kolumna 'date' w formacie zgodnym z korpus_surowy.csv: DD.MM.YYYY, HH:MM

Driver: notebooks/00a_scraping.ipynb.
"""
import csv
import gzip
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime

LISTING_URL = "https://oko.press/najnowsze"
PER_PAGE = 18
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

CSV_COLUMNS = [
    "text",            # naglowek
    "kategoria",       # etykieta z karty (pierwsza taksonomia)
    "kategoria_typ",   # 'category' lub 'topic'
    "date",            # DD.MM.YYYY, HH:MM  (jak w korpus_surowy.csv)
    "date_iso",        # ISO 8601 (pewny do parsowania)
    "typ",             # typ tresci OKO.press (post, micro_analysis, na-zywo...)
    "url",             # pelny adres
    "taksonomie",      # wszystkie taksonomie: "Nazwa(typ) | Nazwa(typ)"
]


def fetch(url, retries=3, timeout=30):
    """Pobiera URL z obsluga gzip, retry i wykladniczym backoffem."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": random.choice(USER_AGENTS),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Encoding": "gzip",
                    "Accept-Language": "pl,en;q=0.8",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return data.decode("utf-8", "strict")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            wait = 2 ** attempt + random.random()
            print(f"    [retry {attempt + 1}/{retries}] {type(e).__name__}: {e} -> czekam {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Nie udalo sie pobrac {url}: {last_err}")


def parse_apollo_state(html):
    m = NEXT_DATA_RE.search(html)
    if not m:
        raise ValueError("Brak bloku __NEXT_DATA__ w HTML")
    data = json.loads(m.group(1))
    return data["props"]["pageProps"]["apolloState"]


def ordered_content_refs(state):
    """Zwraca liste refow artykulow w kolejnosci z ROOT_QUERY (malejaco po dacie)."""
    rq = state.get("ROOT_QUERY", {})
    content_keys = [k for k in rq if k.startswith("content(")]
    if not content_keys:
        return []
    refs = rq[content_keys[0]]
    return [r["__ref"] for r in refs if isinstance(r, dict) and "__ref" in r]


def get_taxonomy_key(content_obj):
    for k in content_obj:
        if k.startswith("metadata_taxonomy"):
            return k
    return None


def extract_row(state, ref):
    c = state.get(ref)
    if not c:
        return None

    title = (c.get("title") or "").strip()
    publish_at = c.get("publish_at")  # np. "2026-06-12T08:38:44" lub None
    ctype = c.get("type") or ""

    # URL ze slug
    seo = c.get("seo_fields") or {}
    slug = seo.get("slug") if isinstance(seo, dict) else None
    url = f"https://oko.press/{slug}" if slug else ""

    # Taksonomie (kategoria/temat). Pierwsza = etykieta widoczna w karcie.
    taxes = []
    tkey = get_taxonomy_key(c)
    if tkey and c.get(tkey):
        for tx in c[tkey]:
            tref = (tx.get("taxonomy") or {}).get("__ref")
            tobj = state.get(tref, {}) if tref else {}
            name = tobj.get("name")
            ttype = tobj.get("type")
            if name:
                taxes.append((name, ttype))

    kategoria = taxes[0][0] if taxes else ""
    kategoria_typ = taxes[0][1] if taxes else ""
    taksonomie = " | ".join(f"{n}({t})" for n, t in taxes)

    # Format daty zgodny z korpus_surowy.csv
    date_pl = ""
    if publish_at:
        try:
            dt = datetime.fromisoformat(publish_at)
            date_pl = dt.strftime("%d.%m.%Y, %H:%M")
        except ValueError:
            date_pl = ""

    return {
        "text": title,
        "kategoria": kategoria,
        "kategoria_typ": kategoria_typ,
        "date": date_pl,
        "date_iso": publish_at or "",
        "typ": ctype,
        "url": url,
        "taksonomie": taksonomie,
        "_publish_at": publish_at,  # pomocnicze, nie zapisywane
    }


def load_existing_urls(path):
    seen = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                u = row.get("url")
                if u:
                    seen.add(u)
    return seen


def scrape_okopress(out_csv=None, since=None, max_pages=None, start_page=1, delay=1.2):
    """Scrapuje OKO.press /najnowsze i dopisuje przyrostowo do CSV.

    Parametry:
        out_csv    : sciezka wyjsciowa (domyslnie config.OKO_CSV)
        since      : "YYYY-MM-DD" lub datetime - zatrzymaj przy starszych artykulach
        max_pages  : maks. liczba stron (None = do konca serwisu)
        start_page : strona poczatkowa (wznowienie)
        delay      : opoznienie miedzy stronami [s]
    Zwraca liczbe nowo dopisanych artykulow.
    """
    import config
    out_csv = str(out_csv or config.OKO_CSV)

    since_dt = None
    if since:
        since_dt = since if isinstance(since, datetime) else datetime.fromisoformat(str(since))

    seen_urls = load_existing_urls(out_csv)
    file_exists = os.path.exists(out_csv) and os.path.getsize(out_csv) > 0
    print(f"Wyjscie: {out_csv}  (juz zapisanych URL: {len(seen_urls)})")
    if since_dt:
        print(f"Zatrzymanie po dacie: {since_dt.date()}")

    f = open(out_csv, "a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    if not file_exists:
        writer.writeheader()

    page = start_page
    total_new = 0
    stop = False
    try:
        while not stop:
            if max_pages and (page - start_page) >= max_pages:
                print(f"Osiagnieto limit {max_pages} stron.")
                break

            url = f"{LISTING_URL}?page={page}" if page > 1 else LISTING_URL
            try:
                html = fetch(url)
                state = parse_apollo_state(html)
                refs = ordered_content_refs(state)
            except Exception as e:
                print(f"[strona {page}] BLAD: {e}")
                break

            if not refs:
                print(f"[strona {page}] brak artykulow -> koniec.")
                break

            page_new = 0
            for ref in refs:
                row = extract_row(state, ref)
                if not row or not row["text"]:
                    continue

                # filtr daty (lista jest malejaca -> mozemy przerwac)
                if since_dt and row["_publish_at"]:
                    try:
                        if datetime.fromisoformat(row["_publish_at"]) < since_dt:
                            stop = True
                            continue
                    except ValueError:
                        pass

                if row["url"] and row["url"] in seen_urls:
                    continue

                writer.writerow(row)
                if row["url"]:
                    seen_urls.add(row["url"])
                page_new += 1
                total_new += 1

            f.flush()
            print(f"[strona {page}] +{page_new} nowych (lacznie nowych: {total_new})")

            if stop:
                print(f"Dotarto do daty granicznej {since_dt.date()} -> stop.")
                break

            page += 1
            time.sleep(delay + random.random() * 0.5)
    except KeyboardInterrupt:
        print("\nPrzerwano przez uzytkownika. Postep zapisany.")
    finally:
        f.close()

    print(f"\nGOTOWE. Dopisano {total_new} artykulow do {out_csv}")
    return total_new
