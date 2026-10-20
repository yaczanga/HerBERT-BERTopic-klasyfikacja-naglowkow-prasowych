# -*- coding: utf-8 -*-
"""
Testy gwarancji metodologicznych, od ktorych zaleza wnioski z badania.

Wszystkie dzialaja na malej probie syntetycznej, wiec nie dotykaja korpusu
i wykonuja sie w sekundach. Sprawdzaja trzy rzeczy: brak wycieku informacji
z przyszlosci przy podziale chronologicznym, rozlacznosc podzbiorow oraz to,
ze czyszczenie naglowkow nie usuwa tresci ani nie gubi rekordow.
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.data import chronological_split, class_distribution, load_and_prepare
from src.utils import clean_headline


@pytest.fixture
def corpus():
    """Cztery lata po jednym naglowku dziennie, trzy etykiety na zmiane."""
    dates = pd.date_range("2022-01-01", "2026-02-04", freq="D")
    return pd.DataFrame({
        "text": [f"naglowek {i}" for i in range(len(dates))],
        "label": [i % 3 for i in range(len(dates))],
        "date_parsed": dates,
    })


# --------------------------------------------------------------- brak wycieku
def test_podzial_jest_chronologiczny(corpus):
    """Kazda data w TRAIN jest wczesniejsza niz kazda w VALID, i tak dalej.

    To jest gwarancja, na ktorej stoi caly wniosek o uogolnianiu na dane
    nowsze niz uczace. Jej zlamanie oznacza wyciek informacji z przyszlosci.
    """
    train, valid, test, _, _ = chronological_split(corpus)
    assert train["date_parsed"].max() < valid["date_parsed"].min()
    assert valid["date_parsed"].max() < test["date_parsed"].min()


def test_podzbiory_sa_rozlaczne_i_pelne(corpus):
    """Zaden naglowek nie trafia do dwoch podzbiorow i zaden nie ginie."""
    train, valid, test, _, _ = chronological_split(corpus)
    assert len(train) + len(valid) + len(test) == len(corpus)
    teksty = [set(d["text"]) for d in (train, valid, test)]
    assert teksty[0] & teksty[1] == set()
    assert teksty[1] & teksty[2] == set()
    assert teksty[0] & teksty[2] == set()


def test_okna_maja_zadana_dlugosc(corpus):
    """TEST to ostatni rok, VALID przedostatni - zgodnie z config.TEST_YEARS."""
    _, valid, test, valid_start, test_start = chronological_split(corpus)
    assert (test_start - valid_start).days in (365, 366)
    assert test["date_parsed"].min() >= test_start
    assert valid["date_parsed"].min() >= valid_start


def test_granice_nie_zaleza_od_kolejnosci_wierszy(corpus):
    """Przetasowanie wejscia nie zmienia zawartosci podzbiorow."""
    a = chronological_split(corpus)
    b = chronological_split(corpus.sample(frac=1, random_state=0).reset_index(drop=True))
    for x, y in zip(a[:3], b[:3]):
        assert set(x["text"]) == set(y["text"])


# ------------------------------------------------------- czyszczenie naglowkow
@pytest.mark.parametrize("surowy, oczekiwany", [
    ("[WIDEO] Premier o budzecie", "Premier o budzecie"),
    ("[SONDAZ] Poparcie spada", "Poparcie spada"),
    ("USA naklada cla", "USA naklada cla"),          # prefiks tresciowy zostaje
    ("Bez dopiskow", "Bez dopiskow"),
])
def test_clean_headline_usuwa_tylko_dopiski(surowy, oczekiwany):
    assert clean_headline(surowy).strip() == oczekiwany


def test_clean_headline_oproznia_naglowek_z_samych_dopiskow():
    """Naglowek zlozony wylacznie z dopiskow schodzi do pustego ciagu.

    Sama funkcja czyszczaca go nie ratuje - przed utrata rekordu zabezpiecza
    dopiero load_and_prepare, co sprawdza osobny test nizej. Ten test pilnuje,
    zeby zabezpieczenie nie zostalo uznane za zbedne.
    """
    for s in ("[WIDEO]", "[ROZMOWA] [WIDEO]"):
        assert clean_headline(s) == ""


def test_potok_nie_traci_naglowka_z_samych_dopiskow(tmp_path):
    """load_and_prepare przywraca oryginal, gdy czyszczenie opróznilo naglowek.

    Bez tego zabezpieczenia pusty tekst wypadlby przy dropna i po cichu
    zmienilby liczebnosci podzbiorow.
    """
    surowy = tmp_path / "korpus.csv"
    pd.DataFrame({
        "text": ["[WIDEO]", "[SONDAZ] Poparcie spada", "Zwykly naglowek"],
        "label": [0, 1, 2],
        "date": ["01.03.2022, 10:00", "02.03.2022, 10:00", "03.03.2022, 10:00"],
    }).to_csv(surowy, index=False)

    wynik = load_and_prepare(surowy)

    assert len(wynik) == 3                      # zaden rekord nie zginal
    assert (wynik["text"].str.len() > 0).all()  # zaden naglowek nie jest pusty
    assert "[WIDEO]" in set(wynik["text"])      # oryginal przywrocony dokladnie


# ------------------------------------------------------------------- rozklady
def test_rozklad_klas_sumuje_sie_do_stu(corpus):
    udzialy = class_distribution(corpus)
    assert sum(u for _, u in udzialy.values()) == pytest.approx(100.0, abs=0.3)
    assert sum(c for c, _ in udzialy.values()) == len(corpus)
