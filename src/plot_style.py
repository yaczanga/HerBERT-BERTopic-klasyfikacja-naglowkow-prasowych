"""
System wizualny projektu - jedyne zrodlo barw i proporcji.

Funkcje rysujace nie dobieraja koloru samodzielnie: siegaja po token opisujacy
role, jaka kolor pelni w danych. Cynober to przedmiot uwagi, blekit to
odniesienie. Dobor palety i wyniki walidacji separacji barw sa w README.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Rola: tozsamosc serii. Trzy barwy Okabe-Ito; czwartej odcienia nie dobieramy.
BLUE = '#006c98'
VERMILION = '#c64e31'
GREEN = '#009e73'

SERIES_1 = BLUE          # odniesienie, norma, grupa kontrolna
SERIES_2 = VERMILION     # przedmiot uwagi, kategoria badana
SERIES_3 = GREEN         # trzecia tozsamosc, zarezerwowana dla wydawcow

# Rola: tozsamosc wydawcy, kolejnoscia etykiet z config.GAZETY, bez
# wartosciowania. Najslabsza para ma separacje CIEDE2000 16,7 wobec progu 8.
PUBLISHER_COLORS = {0: SERIES_1, 1: SERIES_2, 2: SERIES_3}
PUBLISHER_LINESTYLES = {0: '-', 1: '--', 2: ':'}
PUBLISHER_STYLES = {label: {'color': PUBLISHER_COLORS[label],
                            'linestyle': PUBLISHER_LINESTYLES[label]}
                    for label in PUBLISHER_COLORS}

# Drugi kanal dla wykresow powierzchniowych: jasnosci barw OKO.press i RP to
# 111 i 106 na 255, wiec po konwersji do szarosci granica pasow znika.
PUBLISHER_HATCHES = {0: '', 1: '///', 2: '...'}

# Rola: stan (wartosciowanie)
RISK = VERMILION         # blad, zdarzenie negatywne
REFERENCE = BLUE         # trafienie, stan bazowy

# Rola: akcent (jeden wyrozniony element na tle masy)
ACCENT = VERMILION
MUTED = '#7b8f9e'        # masa elementow BEZ tozsamosci; nigdy nie trafia
                         # do legendy jako nazwana kategoria

# Rola: neutralna (tusz nie-danych). Kontrast wobec bieli w komentarzu.
TEXT = '#22292e'            # tytul, etykiety osi             14,74:1
TEXT_SECONDARY = '#5c6b78'  # etykiety wartosci, znaczniki     5,48:1
AXES = '#90a0ad'
GRID = '#dde2e6'
BACKGROUND = '#ffffff'

# Rola: porzadkowa (kwartyle, etapy). Jasny koniec ma kontrast 2,21:1 wobec bieli.
ORDINAL_RAMP = ['#8fb4c7', '#6b9db5', '#3a7694', '#1a5572', '#06324a']

# Rola: sekwencyjna (macierz pomylek). Jasny koniec siega bieli celowo: zero ma
# czytac sie jako puste tlo.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    'project_sequential',
    ['#ffffff', '#c3d6e0', '#7fa8bd', '#3a7694', '#06324a'])

# Rola: dywergencyjna (biegunowosc wokol zera). Blekit ujemne, cynober dodatnie.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    'project_diverging',
    ['#006c98', '#b6cad6', '#f2f2f2', '#e8cdc3', '#c64e31'])

# Szerokosci lamu wydawcy (Elsevier): jedna kolumna, poltorej, dwie.
MM = 1 / 25.4
COLUMN_1 = 90 * MM
COLUMN_1_5 = 140 * MM
COLUMN_2 = 190 * MM

# Proporcje figur - zamknieta lista, osobno dla ekranu i dla druku. Profil
# drukarski zachowuje te same stosunki bokow, zmienia sie tylko szerokosc:
# rysunek powstaje od razu w docelowej szerokosci lamu, wiec skala napisow nie
# zalezy od tego, jak mocno zlozenie go pomniejszy.
_PROPORCJE = {
    'screen': {
        'SMALL': (6.5, 4.0),     # jedna seria, prosty komunikat
        'STANDARD': (9.0, 5.5),  # domyslny
        'WIDE': (12.0, 6.0),     # wiele kategorii na osi poziomej, szeregi
        'SQUARE': (7.5, 7.0),    # macierze
    },
    'print': {
        'SMALL': (COLUMN_1, COLUMN_1 * (4.0 / 6.5)),
        'STANDARD': (COLUMN_1_5, COLUMN_1_5 * (5.5 / 9.0)),
        'WIDE': (COLUMN_2, COLUMN_2 * (6.0 / 12.0)),
        'SQUARE': (COLUMN_1_5, COLUMN_1_5 * (7.0 / 7.5)),
    },
}

# Pismo i grubosc kreski. W druku rysunek jest o okolo dwie piate wezszy niz
# na ekranie, wiec token maleje razem z nim - inaczej napisy zjadlyby wykres.
_PISMO = {
    'screen': {'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
               'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
               'lines.linewidth': 2.0, 'lines.markersize': 6, 'grid.linewidth': 0.8},
    'print': {'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
              'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7.5,
              'lines.linewidth': 1.2, 'lines.markersize': 4, 'grid.linewidth': 0.5},
}

# Kreska i znacznik serii danych. Funkcje rysujace siegaja po token, nie po
# liczbe, zeby profil drukarski zmienial je w jednym miejscu.
_SERIA = {'screen': {'linewidth': 1.8, 'markersize': 3.0},
          'print': {'linewidth': 1.0, 'markersize': 2.2}}

SMALL = _PROPORCJE['screen']['SMALL']
STANDARD = _PROPORCJE['screen']['STANDARD']
WIDE = _PROPORCJE['screen']['WIDE']
SQUARE = _PROPORCJE['screen']['SQUARE']

LINEWIDTH = _SERIA['screen']['linewidth']
MARKERSIZE = _SERIA['screen']['markersize']

LAYOUT = 'screen'        # czynny profil; zmienia go apply_style

SAVE_DPI = 150           # rysunki robocze: README, notebooki
PRINT_DPI = 600          # eksport do zlozenia; wymog wydawcy dla rastra


def font_size(factor: float = 1.0) -> float:
    """Rozmiar pisma wyprowadzony z bazowego.

    Napisy wewnatrz rysunku (podpisy komorek macierzy, adnotacje na osi czasu)
    podaja wielokrotnosc tokenu, nie liczbe punktow, wiec malaja razem z nim
    przy przejsciu na profil drukarski.
    """
    return mpl.rcParams['font.size'] * factor


def panel_grid(n_cols: int = 1, n_rows: int = 1) -> tuple[float, float]:
    """Rozmiar figury dla siatki paneli, wyprowadzony z tokenu WIDE.

    Siatka nie miesci sie w zamknietej liscie rozmiarow, bo rosnie z liczba
    paneli; przypada 6 cali na kolumne. W profilu drukarskim szerokosc jest
    przycieta do dwulamu, bo szersza strona nie ma.
    """
    if n_cols < 1 or n_rows < 1:
        raise ValueError('Siatka musi miec co najmniej jeden panel w kazdym wymiarze.')
    ekran = _PROPORCJE['screen']['WIDE']
    skala = WIDE[0] / ekran[0]
    szerokosc = ekran[0] * max(1.0, n_cols / 2) * skala
    wysokosc = ekran[1] * (0.75 if n_rows == 1 else 0.5 * n_rows) * skala
    if LAYOUT == 'print' and szerokosc > COLUMN_2:
        wysokosc *= COLUMN_2 / szerokosc
        szerokosc = COLUMN_2
    return (szerokosc, wysokosc)


def apply_style(profile: str = 'screen') -> None:
    """Ustawia rcParams. Wywolac raz, przed pierwszym rysunkiem.

    profile='print' przelacza proporcje i pismo na wersje do zlozenia
    w czasopismie; patrz src/figures_publication.py.
    """
    global SMALL, STANDARD, WIDE, SQUARE, LINEWIDTH, MARKERSIZE, LAYOUT

    if profile not in _PROPORCJE:
        raise ValueError(f'Nieznany profil {profile!r}; dozwolone: '
                         f'{sorted(_PROPORCJE)}')
    LAYOUT = profile
    rozmiary = _PROPORCJE[profile]
    SMALL, STANDARD = rozmiary['SMALL'], rozmiary['STANDARD']
    WIDE, SQUARE = rozmiary['WIDE'], rozmiary['SQUARE']
    LINEWIDTH = _SERIA[profile]['linewidth']
    MARKERSIZE = _SERIA[profile]['markersize']

    mpl.rcParams.update({
        'figure.figsize': STANDARD,
        'figure.facecolor': BACKGROUND,
        'axes.facecolor': BACKGROUND,
        'savefig.facecolor': BACKGROUND,
        'savefig.dpi': SAVE_DPI if profile == 'screen' else PRINT_DPI,
        'savefig.bbox': 'tight',

        'font.family': 'DejaVu Sans',
        'axes.titlelocation': 'left',
        'axes.titlepad': 12,
        **_PISMO[profile],

        'text.color': TEXT,
        'axes.titlecolor': TEXT,
        'axes.labelcolor': TEXT_SECONDARY,
        'xtick.color': TEXT_SECONDARY,
        'ytick.color': TEXT_SECONDARY,

        # Ramke nosi tylko os odnoszaca - gorna i prawa nie niosa informacji.
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.edgecolor': AXES,
        'xtick.major.size': 0,
        'ytick.major.size': 0,

        'axes.grid': True,
        'axes.grid.axis': 'y',
        'axes.axisbelow': True,
        'grid.color': GRID,

        'legend.frameon': False,
        'axes.prop_cycle': mpl.cycler(color=[BLUE, VERMILION, GREEN]),
    })
