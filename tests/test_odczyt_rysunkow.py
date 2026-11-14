# -*- coding: utf-8 -*-
"""Gwarancje odczytu punktow danych z zapisanych rysunkow.

Odczyt zastepuje dane zrodlowe tylko tam, gdzie ich nie ma, i tylko wtedy ma
sens, gdy zgadza sie z liczbami podanymi w artykule. Testy sprawdzaja wlasnie
zgodnosc, a nie sam przebieg odczytu: liczby kontrolne - sumy naglowkow na
wydawce i statystyki okresu walidacyjnego - nie wchodza do odczytu w zaden
sposob, wiec trafienie w nie jest niezaleznym potwierdzeniem.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src import figures_digitised as odczyt


@pytest.fixture(scope="module")
def tabele():
    return {nazwa: odczyt.wczytaj(nazwa) for nazwa in odczyt.RYSUNKI}


def test_odczyt_zgadza_sie_z_liczbami_z_artykulu(tabele):
    """Prog 0,5% jest o rzad wielkosci luzniejszy niz zmierzony blad."""
    kontrola = odczyt.sprawdz(tabele)
    najgorsza = kontrola.loc[kontrola["blad"].abs().idxmax()]
    assert kontrola["blad"].abs().max() < 0.005, najgorsza.to_dict()


def test_odczyt_z_png_odtwarza_zapisane_tabele(tabele):
    """Tabele w outputs/digitised/ maja pochodzic z rysunkow w repozytorium."""
    for nazwa, zapisana in tabele.items():
        swiezy = odczyt.odczytaj(nazwa)
        pd.testing.assert_frame_equal(swiezy, zapisana, check_dtype=False,
                                      check_freq=False, atol=1e-4)


def test_tabele_obejmuja_caly_okres(tabele):
    for tab in tabele.values():
        assert len(tab) == 50
        assert tab.index[0] == pd.Timestamp("2022-01-01")
        assert tab.index[-1] == pd.Timestamp("2026-02-01")


def test_liczebnosci_sa_liczbami_calkowitymi(tabele):
    licz = tabele["liczebnosci"]
    assert (licz.to_numpy() == licz.to_numpy().astype(int)).all()
    assert (licz.to_numpy() > 0).all()


def test_korpus_zastepczy_ma_ksztalt_korpusu():
    """plot_monthly_counts zlicza nagłowki sam, wiec dostaje wiersz na naglowek."""
    korpus = odczyt.korpus_zastepczy()
    assert set(korpus.columns) == {"date_parsed", "label"}
    assert len(korpus) == int(odczyt.wczytaj("liczebnosci").to_numpy().sum())
    assert set(korpus["label"]) == {0, 1, 2}
    # Krance zakresu ustawiaja granice osi poziomej.
    assert korpus["date_parsed"].min() == odczyt.KORPUS_OD
    assert korpus["date_parsed"].max() == odczyt.KORPUS_DO


def test_pewnosc_zastepcza_jest_w_postaci_dlugiej():
    pewnosc = odczyt.pewnosc_zastepcza()
    assert list(pewnosc.columns) == ["date", "label", "confidence"]
    assert len(pewnosc) == 150
    assert pewnosc["confidence"].between(0.5, 0.9).all()
