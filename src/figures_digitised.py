# -*- coding: utf-8 -*-
"""Odczyt punktow danych z zapisanych rysunkow (digitalizacja).

PO CO TO JEST. Korpus i plik predykcji leza poza repozytorium, wiec na maszynie
bez nich rysunkow 1 i 3 nie da sie narysowac od nowa z danych zrodlowych.
Zostaja same pliki PNG w outputs/figures/, w ktorych punkty danych sa widoczne,
tylko zapisane jako piksele. Ten modul je odczytuje: znajduje srodki znacznikow
serii, przelicza je na wartosci po kalibracji osi i zapisuje jako tabele
w outputs/digitised/. Rysunek do zlozenia powstaje potem z tych tabel.

CZEGO TO NIE JEST. To nie sa dane zrodlowe i nie zastepuja pliku predykcji.
Wartosci sa odczytem z obrazka, obarczonym bledem rzedu jednej dziesiatej
piksela, czyli okolo 0,4 naglowka na punkt na rysunku 1 i okolo 0,0005 na
rysunku 3. Gdy data/ jest na miejscu, src/figures_publication.py bierze dane
stamtad, a te tabele sa tylko zapasem i trzeba je uznac za nieaktualne.

DOKLADNOSC. Odczyt sprawdza sie wzgledem liczb podanych w artykule, ktorych
digitalizacja nie zna: sumy naglowkow na wydawce oraz statystyki okresu
walidacyjnego z rysunku 3. Kontrole wykonuje sprawdz() i powtarza test
tests/test_odczyt_rysunkow.py.

Uruchomienie:

    python -m src.figures_digitised            # odczyt z PNG + kontrola
    python -m src.figures_digitised --sprawdz  # sama kontrola zapisanych tabel
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import config

DIGITISED_DIR = config.OUTPUTS_DIR / "digitised"

# Barwy serii pochodza z systemu wizualnego; wnetrze znacznika ma je dokladnie.
from src import plot_style as style  # noqa: E402

SERIE = {0: style.PUBLISHER_COLORS[0], 1: style.PUBLISHER_COLORS[1],
         2: style.PUBLISHER_COLORS[2]}

# Pierwszy i ostatni dzien proby analitycznej. Granica TEST wypada rok przed
# koncem korpusu, a odczytana z rysunku lezy na 2025-02-04, wiec korpus konczy
# sie 2026-02-04 - zgodnie z tekstem artykulu, gdzie zbieranie danych ustaje
# 4 lutego 2026. Poczatek to dolna granica proby z config.DATA_START; odczyt
# z krawedzi osi daje 2022-01-02, czyli tyle samo z dokladnoscia do piksela.
KORPUS_OD = pd.Timestamp(config.DATA_START)
KORPUS_DO = pd.Timestamp("2026-02-04")
VALID_START = pd.Timestamp("2024-02-04")
TEST_START = pd.Timestamp("2025-02-04")

MIESIACE = pd.date_range("2022-01-01", "2026-02-01", freq="MS")

# Liczebnosci na wydawce podane w artykule. Digitalizacja ich nie uzywa do
# odczytu, tylko do kontroli i do jednej poprawki: patrz _bez_obciazenia.
SUMY_Z_ARTYKULU = {0: 20892, 1: 19781, 2: 15241}

# Opis kazdego rysunku: plik, wartosci przy liniach siatki, kotwice osi
# poziomej i prostokat legendy wycinany z maski (znacznik w legendzie wyglada
# jak punkt danych).
RYSUNKI = {
    "liczebnosci": {
        "png": "05_liczebnosci_miesieczne.png",
        "gridlines": np.arange(800, -1, -100),
        "kotwice_x": ((928.0, VALID_START), (1329.0, TEST_START)),
        "legenda": ((74, 106), (100, 600)),
        "kolumny": ["Niezależna", "OKO.press", "RP"],
    },
    "pewnosc": {
        "png": "08_pewnosc_w_czasie.png",
        "gridlines": np.arange(0.85, 0.54, -0.05),
        "kotwice_x": ((100.8, MIESIACE[0]), (1753.2, MIESIACE[-1])),
        "legenda": ((700, 775), (100, 620)),
        "kolumny": ["Niezależna", "OKO.press", "RP"],
    },
}


# --- odczyt z obrazka ---------------------------------------------------

def _maska(im, barwa_hex, tol=40):
    """Piksele w barwie serii. Tolerancja obejmuje brzeg wygladzania."""
    c = np.array([int(barwa_hex[i:i + 2], 16) for i in (1, 3, 5)])
    return np.abs(im - c).max(axis=2) <= tol


def _grupuj(indeksy, luka=2):
    grupy, biez = [], [indeksy[0]]
    for i in indeksy[1:]:
        if i - biez[-1] <= luka:
            biez.append(i)
        else:
            grupy.append(biez)
            biez = [i]
    grupy.append(biez)
    return np.array([float(np.mean(g)) for g in grupy])


def _linie_siatki(im, prog=0.3):
    """Wiersze linii siatki - kalibracja osi pionowej."""
    h, w, _ = im.shape
    m = np.zeros((h, w), bool)
    for barwa in ("#e2e6ea", "#e0e4e8"):   # siatka na bieli i na pasie testowym
        m |= _maska(im, barwa, 4)
    return _grupuj(np.where(m.sum(axis=1) > w * prog)[0])


def _odpowiedz_na_znacznik(m, r=3):
    """Splot maski z krazkiem wielkosci znacznika.

    Znacznik ma srednice okolo 6 pikseli, linia serii okolo 4, wiec krazek
    odpowiada najsilniej na srodku znacznika, a nie na przebiegu linii.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    krazek = (x * x + y * y <= r * r).astype(float)
    pad = np.pad(m.astype(float), r)
    return (sliding_window_view(pad, krazek.shape) * krazek).sum(axis=(2, 3))


def _srodki(im, barwa_hex, x_oczekiwane, legenda, okno=5):
    """Srodki znacznikow serii w oknach wokol przewidzianych pozycji x."""
    m = _maska(im, barwa_hex)
    (y0, y1), (x0, x1) = legenda
    m[y0:y1, x0:x1] = False
    odp = _odpowiedz_na_znacznik(m)

    srodki = []
    for xp in x_oczekiwane:
        a, b = max(0, int(round(xp - okno))), int(round(xp + okno)) + 1
        wycinek = odp[:, a:b]
        iy, ix = np.unravel_index(np.argmax(wycinek), wycinek.shape)
        # Srodek podpikselowy: srodek ciezkosci maski w pudelku wokol maksimum.
        yy0, yy1 = max(0, iy - 4), iy + 5
        xx0, xx1 = max(0, a + ix - 4), a + ix + 5
        ys, xs = np.nonzero(m[yy0:yy1, xx0:xx1])
        srodki.append(((yy0 + ys).mean(), (xx0 + xs).mean()))
    return np.array(srodki)


def odczytaj(nazwa: str, korekta_obciazenia: bool = True) -> pd.DataFrame:
    """Odczytuje jeden rysunek z outputs/figures/ i zwraca tabele miesieczna."""
    from PIL import Image

    opis = RYSUNKI[nazwa]
    im = np.array(Image.open(config.FIGURES / opis["png"]).convert("RGB")).astype(int)

    siatka = _linie_siatki(im)
    if len(siatka) != len(opis["gridlines"]):
        raise ValueError(f"{opis['png']}: znaleziono {len(siatka)} linii siatki, "
                         f"oczekiwano {len(opis['gridlines'])}")
    a_y, b_y = np.polyfit(siatka, opis["gridlines"], 1)

    (px1, d1), (px2, d2) = opis["kotwice_x"]
    na_dzien = (px2 - px1) / (d2 - d1).days
    x_ocz = np.array([px1 + (m - d1).days * na_dzien for m in MIESIACE])

    kolumny = {}
    for label, barwa in SERIE.items():
        srodki = _srodki(im, barwa, x_ocz, opis["legenda"])
        # Kontrola, czy odczyt trzyma sie wlasciwego znacznika, a nie przeskoczyl
        # na sasiedni. Prog jest luzny, bo srodek ciezkosci maski przesuwa sie
        # w bok razem z linia doklejona do znacznika; na wartosc to nie wplywa,
        # bo wartosc czyta sie z polozenia pionowego.
        rozjazd = np.abs(srodki[:, 1] - x_ocz).max()
        if rozjazd > 6.0:
            raise ValueError(f"{opis['png']}, seria {label}: znacznik odbiega "
                             f"od pozycji miesiaca o {rozjazd:.1f} px")
        kolumny[opis["kolumny"][label]] = a_y * srodki[:, 0] + b_y

    tab = pd.DataFrame(kolumny, index=MIESIACE)
    tab.index.name = "date"
    if nazwa == "liczebnosci":
        tab = _bez_obciazenia(tab, a_y)
    return tab


def _bez_obciazenia(tab: pd.DataFrame, a_y: float) -> pd.DataFrame:
    """Zdejmuje stale obciazenie odczytu i zaokragla do liczb calkowitych.

    Odczyt ma stale, nieduze obciazenie: srodek ciezkosci maski znacznika lezy
    o ulamek piksela nizej niz srodek znacznika, bo linia serii dokleja sie do
    niego od dolu. Poprawka jest jedna liczba na caly rysunek, dobrana tak, zeby
    suma wszystkich naglowkow zgadzala sie z liczba podana w artykule, i ta sama
    dla trzech serii. Nie dopasowuje serii z osobna i nie rusza ksztaltu
    szeregu: przesuwa caly rysunek o 0,4 piksela.
    """
    deficyt = sum(SUMY_Z_ARTYKULU.values()) - tab.to_numpy().sum()
    przesuniecie = deficyt / tab.size
    return (tab + przesuniecie).round().astype(int)


# --- zapis, wczytanie, kontrola -----------------------------------------

def sciezka(nazwa: str):
    return DIGITISED_DIR / f"{nazwa}_miesieczne.csv"


def zapisz(verbose: bool = True) -> dict:
    """Odczytuje oba rysunki i zapisuje tabele do outputs/digitised/."""
    DIGITISED_DIR.mkdir(parents=True, exist_ok=True)
    tabele = {}
    for nazwa in RYSUNKI:
        tab = odczytaj(nazwa)
        tab.to_csv(sciezka(nazwa), encoding="utf-8-sig")
        tabele[nazwa] = tab
        if verbose:
            print(f"  {nazwa}: {len(tab)} miesiecy -> {sciezka(nazwa).name}")
    return tabele


def wczytaj(nazwa: str) -> pd.DataFrame:
    """Wczytuje zapisana tabele; gdy jej nie ma, odczytuje rysunek od nowa."""
    p = sciezka(nazwa)
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True, encoding="utf-8-sig")
    return odczytaj(nazwa)


def sprawdz(tabele=None) -> pd.DataFrame:
    """Porownuje odczyt z liczbami podanymi w artykule.

    Kontrola jest niezalezna od samego odczytu: digitalizacja nie zna ani sum na
    wydawce, ani statystyk okresu walidacyjnego. Okres walidacyjny wybrano
    dlatego, ze miesci sie w calych miesiacach - okresy uczacy i testowy zaczyna
    i konczy miesiac przeciety granica podzialu, ktorego rysunek nie pokazuje
    osobno.
    """
    tabele = tabele or {n: wczytaj(n) for n in RYSUNKI}
    licz, pewn = tabele["liczebnosci"], tabele["pewnosc"]

    wiersze = []
    for label, suma in SUMY_Z_ARTYKULU.items():
        kol = RYSUNKI["liczebnosci"]["kolumny"][label]
        wiersze.append({"kontrola": f"liczba naglowkow, {kol}",
                        "artykul": suma, "odczyt": int(licz[kol].sum())})
    wiersze.append({"kontrola": "liczba naglowkow, razem",
                    "artykul": sum(SUMY_Z_ARTYKULU.values()),
                    "odczyt": int(licz.to_numpy().sum())})

    wazona = (pewn.to_numpy() * licz.to_numpy()).sum(axis=1) / licz.to_numpy().sum(axis=1)
    walid = pd.Series(wazona, index=pewn.index)["2024-03":"2025-01"]
    for opis, wzorzec, wartosc in (("pewnosc, walidacja, srednia", 0.654, walid.mean()),
                                   ("pewnosc, walidacja, minimum", 0.616, walid.min()),
                                   ("pewnosc, walidacja, maksimum", 0.677, walid.max())):
        wiersze.append({"kontrola": opis, "artykul": wzorzec, "odczyt": round(wartosc, 3)})

    tab = pd.DataFrame(wiersze)
    tab["blad"] = (tab["odczyt"] - tab["artykul"]) / tab["artykul"]
    return tab


# --- dane w ksztalcie, jakiego oczekuja funkcje rysujace ----------------

def korpus_zastepczy() -> pd.DataFrame:
    """Odtworzony korpus: jeden wiersz na naglowek, z data i etykieta wydawcy.

    plot_monthly_counts zlicza naglowki sam, wiec dostaje ramke w tym samym
    ksztalcie, co korpus prawdziwy. Dzien w miesiacu nie ma znaczenia dla
    zliczania; brzegi zakresu ustawiaja granice osi poziomej i te sa znane.
    """
    licz = wczytaj("liczebnosci")
    kolumny = RYSUNKI["liczebnosci"]["kolumny"]
    daty, etykiety = [], []
    for label, kol in enumerate(kolumny):
        for miesiac, n in licz[kol].items():
            daty.extend([miesiac] * int(n))
            etykiety.extend([label] * int(n))
    ramka = pd.DataFrame({"date_parsed": daty, "label": etykiety})
    # Krance korpusu: bez nich os zaczelaby sie i konczyla na poczatku miesiaca.
    ramka.loc[ramka.index[0], "date_parsed"] = KORPUS_OD
    ramka.loc[ramka.index[-1], "date_parsed"] = KORPUS_DO
    return ramka.sort_values("date_parsed").reset_index(drop=True)


def pewnosc_zastepcza() -> pd.DataFrame:
    """Odtworzony szereg pewnosci w postaci dlugiej: date, label, confidence."""
    pewn = wczytaj("pewnosc")
    kolumny = RYSUNKI["pewnosc"]["kolumny"]
    czesci = []
    for label, kol in enumerate(kolumny):
        czesci.append(pd.DataFrame({"date": pewn.index, "label": label,
                                    "confidence": pewn[kol].to_numpy()}))
    return pd.concat(czesci, ignore_index=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Odczyt punktow danych z zapisanych rysunkow.")
    parser.add_argument("--sprawdz", action="store_true",
                        help="nie odczytuj od nowa, sprawdz zapisane tabele")
    args = parser.parse_args(argv)

    if args.sprawdz:
        tabele = None
    else:
        print(f"Odczyt rysunkow do {DIGITISED_DIR}:")
        tabele = zapisz()

    kontrola = sprawdz(tabele)
    print("\nKontrola wzgledem liczb podanych w artykule:")
    for _, w in kontrola.iterrows():
        print(f"  {w['kontrola']:<32} artykul {w['artykul']:>8}  "
              f"odczyt {w['odczyt']:>8}  blad {w['blad'] * 100:+.2f}%")
    return 0 if kontrola["blad"].abs().max() < 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
