# -*- coding: utf-8 -*-
"""Gwarancje eksportu rysunkow do zlozenia w czasopismie.

Wydawca stawia trzy warunki, ktorych nie widac w samym kodzie rysujacym:
osobny plik na rysunek, brak podpisu wrysowanego w obrazek i rozdzielczosc
rastra. Testy pilnuja wlasnie ich, a przy okazji tego, ze profil drukarski nie
zmienia rysunkow roboczych.

Macierz pomylek odrysowuje sie z zapisanej tabeli liczebnosci, wiec testy nie
potrzebuja ani korpusu, ani pliku predykcji.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import pytest

from src import plot_style as style

# Tytul lezy po lewej (rcParams 'axes.titlelocation'), wiec get_title() bez
# argumentu, czyli dla srodka, zawsze zwraca pusty napis.
POLOZENIA = ("left", "center", "right")


def tytul_osi(ax):
    return "".join(ax.get_title(loc=gdzie) for gdzie in POLOZENIA)


@pytest.fixture(autouse=True)
def profil_ekranowy():
    """Przywraca profil ekranowy po tescie - jest ustawieniem globalnym."""
    yield
    style.apply_style("screen")


@pytest.fixture
def eksport():
    from src import figures_publication

    return figures_publication


# --- tytuly -------------------------------------------------------------

def test_brak_tytulu_gdy_wylaczony():
    from src.visualization import _tekst_tytulu

    assert _tekst_tytulu(None, "domyslny") == "domyslny"
    assert _tekst_tytulu(False, "domyslny") == ""
    assert _tekst_tytulu("", "domyslny") == ""
    assert _tekst_tytulu("wlasny", "domyslny") == "wlasny"


def test_rysunek_do_druku_nie_ma_tytulu_na_obrazku(eksport):
    """Podpis rysunku nalezy do tekstu artykulu, nie do pliku graficznego."""
    import matplotlib.pyplot as plt

    style.apply_style("print")
    fig = eksport.fig2_macierz_pomylek()
    try:
        assert fig._suptitle is None or not fig._suptitle.get_text()
        for ax in fig.axes:
            assert tytul_osi(ax) == ""
    finally:
        plt.close(fig)


def test_rysunek_roboczy_zachowuje_tytul():
    """Domyslne wywolanie, to z notebookow, nadal podpisuje rysunek."""
    import matplotlib.pyplot as plt

    from src.visualization import plot_confusion_matrix

    fig = plot_confusion_matrix([0, 1, 2], [0, 1, 2])
    try:
        assert tytul_osi(fig.axes[0]) == "Confusion matrix, test set"
    finally:
        plt.close(fig)


# --- profil drukarski ---------------------------------------------------

def test_profil_drukarski_daje_szerokosc_lamu():
    style.apply_style("print")
    assert style.WIDE[0] == pytest.approx(style.COLUMN_2)
    assert style.SQUARE[0] == pytest.approx(style.COLUMN_1_5)
    # Dwulam wydawcy to 190 mm.
    assert style.COLUMN_2 * 25.4 == pytest.approx(190.0)


def test_profil_drukarski_zachowuje_proporcje():
    """Zmienia sie szerokosc, nie ksztalt - inaczej rysunek zmienilby wymowe."""
    ekran = {nazwa: getattr(style, nazwa)
             for nazwa in ("SMALL", "STANDARD", "WIDE", "SQUARE")}
    style.apply_style("print")
    for nazwa, (szer_e, wys_e) in ekran.items():
        szer_d, wys_d = getattr(style, nazwa)
        assert wys_d / szer_d == pytest.approx(wys_e / szer_e)


def test_profil_ekranowy_jest_nietkniety():
    """Rysunki w repozytorium maja wyjsc tak samo jak przed dodaniem druku."""
    style.apply_style("screen")
    assert style.WIDE == (12.0, 6.0)
    assert style.SQUARE == (7.5, 7.0)
    assert style.LINEWIDTH == 1.8
    assert style.MARKERSIZE == 3.0
    assert style.font_size(12 / 11) == pytest.approx(12.0)


def test_nieznany_profil_jest_bledem():
    with pytest.raises(ValueError):
        style.apply_style("plakat")


# --- pliki wynikowe -----------------------------------------------------

def test_kazdy_rysunek_trafia_do_osobnego_pliku(tmp_path, eksport):
    wynik = eksport.build(nazwy=["Fig2"], out_dir=tmp_path,
                          formats=("pdf", "tif"), dpi=600, verbose=False)
    assert set(wynik) == {"Fig2"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Fig2.pdf", "Fig2.tif"]


def test_raster_ma_zadana_rozdzielczosc_i_trzy_kanaly(tmp_path, eksport):
    """600 dpi z zapasem wobec progu 500 dpi dla kreski z rastrem.

    Kanal alfa jest zdejmowany: sklad czesto renderuje przezroczystosc na
    czarno.
    """
    from PIL import Image

    eksport.build(nazwy=["Fig2"], out_dir=tmp_path, formats=("tif",),
                  dpi=600, verbose=False)
    with Image.open(tmp_path / "Fig2.tif") as im:
        assert im.info["dpi"] == (600, 600)
        assert im.mode == "RGB"
        # Szerokosc nie moze zejsc ponizej lamu jednokolumnowego (90 mm).
        assert im.size[0] / 600 * 25.4 > 90


def test_nieznana_nazwa_rysunku_jest_bledem(tmp_path, eksport):
    with pytest.raises(ValueError):
        eksport.build(nazwy=["Fig9"], out_dir=tmp_path, verbose=False)


def test_macierz_odtworzona_z_tabeli_zgadza_sie_z_artykulem(eksport):
    """Tabela liczebnosci jest zrodlem Fig2, gdy nie ma pliku predykcji."""
    import numpy as np

    y_true, y_pred = eksport._etykiety_testu()
    assert len(y_true) == len(y_pred)
    trafienia = (np.asarray(y_true) == np.asarray(y_pred)).sum()
    # Liczby z artykulu: 12 807 naglowkow testowych, 9 765 trafien.
    assert len(y_true) == 12807
    assert trafienia == 4284 + 3423 + 2058
