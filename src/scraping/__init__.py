# -*- coding: utf-8 -*-
"""Scrapery naglowkow. Driver: notebooks/00a_scraping.ipynb."""
from src.scraping.scrape_newspapers import scrape_sources
from src.scraping.scrape_okopress import scrape_okopress

NEWSPAPER_SOURCES = ["niezalezna", "rp"]
ALL_SOURCES = NEWSPAPER_SOURCES + ["oko"]

__all__ = ["scrape_sources", "scrape_okopress", "NEWSPAPER_SOURCES", "ALL_SOURCES"]
