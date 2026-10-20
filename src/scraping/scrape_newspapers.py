# -*- coding: utf-8 -*-
"""
Asynchroniczny scraper naglowkow: Niezalezna (label 0) oraz RP (label 2).
Driver: notebooks/00a_scraping.ipynb.

Klase 1 w aktualnym pipelinie stanowi OKO.press (zob. src/scraping/scrape_okopress).
"""
import asyncio
import random
import time

LIMIT_POLACZEN = 30  # max rownoczesnych zadan

HEADERS_NORMAL = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
}


async def fetch(session, url, headers=None, retries=2):
    headers = headers or HEADERS_NORMAL
    for i in range(retries + 1):
        try:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            async with session.get(url, headers=headers, timeout=15) as response:
                if response.status == 200:
                    return await response.text()
                if response.status in (403, 429, 503):
                    await asyncio.sleep(2 * (i + 1))
                else:
                    return None
        except Exception:
            if i == retries:
                return None
            await asyncio.sleep(1)
    return None


# --- Niezalezna (label 0) ---
async def _article_niezalezna(session, url, sem):
    from bs4 import BeautifulSoup
    async with sem:
        html = await fetch(session, url, headers=HEADERS_NORMAL)
        if not html:
            return None
        topper = BeautifulSoup(html, "html.parser").find(id="article-topper")
        if topper:
            t = topper.find("time")
            if t:
                return t.get_text(strip=True)
        return None


async def scrape_niezalezna(max_pages):
    from bs4 import BeautifulSoup
    import aiohttp
    print("--- Niezalezna ---")
    cat, base = "https://niezalezna.pl/polityka", "https://niezalezna.pl"
    data, sem = [], asyncio.Semaphore(LIMIT_POLACZEN)
    async with aiohttp.ClientSession() as session:
        for i in range(1, max_pages + 1):
            print(f"Niezalezna [{i}/{max_pages}]")
            list_html = await fetch(session, f"{cat}?page={i}")
            if not list_html:
                continue
            links = BeautifulSoup(list_html, "html.parser").select("div[x-data] a.block")
            if not links:
                break
            tasks, titles = [], []
            for a in links:
                href = a.get("href")
                img = a.find("img")
                title = img.get("alt") if img else a.get_text(strip=True)
                if title and len(title) > 20 and href:
                    full = href if href.startswith("http") else base + href
                    titles.append(title)
                    tasks.append(_article_niezalezna(session, full, sem))
            for title, date in zip(titles, await asyncio.gather(*tasks)):
                data.append({"text": title.strip(), "label": 0, "date": date})
    return data


# --- RP (label 2) ---
async def _article_rp(session, url, sem):
    from bs4 import BeautifulSoup
    async with sem:
        html = await fetch(session, url, headers=HEADERS_NORMAL)
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        span = soup.find("span", id="livePublishedAtContainer")
        if span:
            return span.get_text(strip=True)
        meta = soup.find("meta", property="article:published_time")
        return meta.get("content") if meta else None


async def scrape_rp(max_pages):
    from bs4 import BeautifulSoup
    import aiohttp
    print("--- RP.pl ---")
    tmpl, base = "https://www.rp.pl/wydarzenia/kraj/polityka/{}", "https://www.rp.pl"
    data, sem = [], asyncio.Semaphore(LIMIT_POLACZEN)
    async with aiohttp.ClientSession() as session:
        for i in range(1, max_pages + 1):
            print(f"RP.pl [{i}/{max_pages}]")
            list_html = await fetch(session, tmpl.format(i))
            if not list_html:
                continue
            links = BeautifulSoup(list_html, "html.parser").select(
                "h2.teaser__title a, h3.teaser__title a, h2.teaser--title a, a.contentLink")
            if not links:
                break
            tasks, titles, seen = [], [], set()
            for a in links:
                text = a.get_text(strip=True)
                href = a.get("href")
                if text and len(text) > 20 and href and "prenumerat" not in text.lower():
                    full = href if href.startswith("http") else base + href
                    if full not in seen:
                        seen.add(full)
                        titles.append(text)
                        tasks.append(_article_rp(session, full, sem))
            for title, date in zip(titles, await asyncio.gather(*tasks)):
                data.append({"text": title.strip(), "label": 2, "date": date})
    return data


# Rejestr zrodel gazetowych: nazwa -> (funkcja, mnoznik stron)
SCRAPERS = {
    "niezalezna": (scrape_niezalezna, 2),
    "rp": (scrape_rp, 2),
}


async def scrape_sources(sources, pages=1000, out_csv=None):
    """Scrapuje wybrane zrodla gazetowe i zapisuje do jednego CSV.

    sources : lista nazw z SCRAPERS (np. ["niezalezna", "rp"])
    pages   : bazowa liczba stron listy (mnozona przez wspolczynnik zrodla)
    out_csv : plik wyjsciowy (domyslnie config.NEWSPAPERS_CSV)
    """
    import pandas as pd
    import config
    out_csv = out_csv or config.NEWSPAPERS_CSV
    start = time.time()
    res = []
    for s in sources:
        fn, mult = SCRAPERS[s]
        res += await fn(pages * mult)
    if not res:
        print("Brak danych.")
        return None
    df = pd.DataFrame(res).drop_duplicates(subset=["text"]).sample(frac=1).reset_index(drop=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nSUKCES: {len(df)} rekordow w {time.time() - start:.1f}s -> {out_csv}")
    print(df["label"].value_counts())
    return df

