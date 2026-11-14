# -*- coding: utf-8 -*-
"""Eksport rysunkow do zlozenia w czasopismie.

Wydawca (Elsevier, "Information Economics and Policy") przyjmuje rysunki jako
osobne pliki: jeden plik na rysunek, nazwany numerem z tekstu (Fig1, Fig2,
...), w szerokosci lamu, bez podpisu wrysowanego w obrazek - podpis nalezy do
tekstu artykulu - i w rozdzielczosci nie nizszej niz 300 dpi dla rastra, 500
dpi dla rysunku laczacego kreske z rastrem, 1000 dpi dla samej kreski.

Modul nie rysuje niczego od nowa: wola te same funkcje z src/visualization.py,
co notebooki, tyle ze w profilu drukarskim (plot_style.apply_style('print'))
i z title=False. Dzieki temu rysunek zlozony w czasopismie i rysunek
w repozytorium pokazuja te same dane, narysowane tym samym kodem.

Numeracja Fig1-Fig3 jest numeracja z tekstu artykulu:

    Fig1  miesieczne liczebnosci naglowkow wedlug wydawcy
    Fig2  macierz pomylek na zbiorze testowym, udzialy w wierszu
    Fig3  srednie prawdopodobienstwo klasy prawdziwej wedlug wydawcy i miesiaca

Uruchomienie z korzenia projektu:

    python -m src.figures_publication                 # Fig1-Fig3
    python -m src.figures_publication --all           # i pozostale rysunki
    python -m src.figures_publication --formats pdf,tif,png
    python -m src.figures_publication --dpi 1000

Fig1 i Fig3 czytaja korpus i plik predykcji z data/, ktorych repozytorium nie
zawiera (patrz README, sekcja "Dane"). Bez nich powstaja z odczytu zapisanych
PNG (src/figures_digitised.py), a eksport mowi o tym w komunikacie - to wyjscie
awaryjne, nie rownowazne zrodlo. Fig2 nie potrzebuje ani jednego, ani drugiego:
odrysowuje sie z tabeli outputs/tabela_macierz_pomylek_test.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

import config
from src import plot_style as style

# Katalog eksportu trzymamy osobno od rysunkow roboczych: pliki do zlozenia maja
# inne rozmiary i inna numeracje, a mieszanie jednych z drugimi konczy sie
# wyslaniem do wydawcy rysunku z tytulem.
PUBLICATION_DIR = config.FIGURES / "publication"

# Domyslnie wektor i raster. PDF jest wektorowy, wiec nie ma go czym "podniesc"
# rozdzielczoscia; TIFF z kompresja LZW to format bezstratny, ktorego wymaga
# wydawca dla rastra.
DEFAULT_FORMATS = ("pdf", "tif")
DEFAULT_DPI = style.PRINT_DPI


def _sciezka(stem: str, ext: str, out_dir: Path) -> Path:
    return out_dir / f"{stem}.{ext}"


def zapisz(fig, stem: str, out_dir: Path = PUBLICATION_DIR,
           formats=DEFAULT_FORMATS, dpi: int = DEFAULT_DPI) -> list[Path]:
    """Zapisuje jeden rysunek pod nazwa stem w kazdym z podanych formatow.

    Plik jest kadrowany do zawartosci. Szerokosc lamu z profilu drukarskiego
    rzadzi proporcja pisma do wykresu, nie zas ostatecznym rozmiarem pliku:
    po odcieciu marginesu rysunek bywa o kilka milimetrow wezszy, a sklad
    i tak skaluje go do lamu. Zostawienie marginesu daloby biale pasy, bo
    macierz pomylek ma zablokowane proporcje komorki.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    zapisane = []
    with mpl.rc_context({"savefig.bbox": "tight"}):
        for ext in formats:
            sciezka = _sciezka(stem, ext, out_dir)
            kwargs = {}
            if ext in ("tif", "tiff"):
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(sciezka, dpi=dpi, facecolor=style.BACKGROUND,
                        pad_inches=0.02, **kwargs)
            if ext in ("tif", "tiff"):
                _splaszcz_tiff(sciezka, dpi)
            zapisane.append(sciezka)
    return zapisane


# Rysunki zlozone z odczytu obrazka, a nie z danych zrodlowych. Lista sluzy do
# tego, zeby eksport powiedzial o tym na koniec, a nie tylko w locie.
ODTWORZONE = []


def _ostrzez(stem: str, brakujacy) -> None:
    ODTWORZONE.append(stem)
    print(f"  UWAGA {stem}: brak {brakujacy.name}, rysunek odtworzony z odczytu "
          f"zapisanego PNG (src/figures_digitised.py)")


def _splaszcz_tiff(sciezka: Path, dpi: int) -> None:
    """Zdejmuje kanal alfa z zapisanego TIFF-a i wkleja rysunek na biel.

    Matplotlib zapisuje raster z kanalem alfa. Systemy skladu czesto go nie
    czytaja i renderuja przezroczystosc na czarno, wiec plik oddawany wydawcy
    ma miec trzy kanaly i tlo wtopione na stale.
    """
    from PIL import Image

    with Image.open(sciezka) as im:
        if im.mode not in ("RGBA", "LA", "P"):
            return
        maska = im.convert("RGBA").split()[-1]
        plaski = Image.new("RGB", im.size, style.BACKGROUND)
        plaski.paste(im.convert("RGB"), mask=maska)
    plaski.save(sciezka, compression="tiff_lzw", dpi=(dpi, dpi))


def _wymiary(sciezki, dpi: int) -> str:
    """Rzeczywisty rozmiar zapisanego rastra, odczytany z pliku."""
    from PIL import Image

    rastry = [s for s in sciezki if s.suffix.lower() in (".tif", ".tiff", ".png")]
    if not rastry:
        return f"{dpi} dpi"
    with Image.open(rastry[0]) as im:
        szer, wys = im.size
    return (f"{szer / dpi * 25.4:.0f}x{wys / dpi * 25.4:.0f} mm, "
            f"{szer}x{wys} px, {dpi} dpi")


# --- poszczegolne rysunki -------------------------------------------------
# Kazda funkcja zwraca gotowa figure albo podnosi FileNotFoundError, gdy jej
# danych nie ma w repozytorium. Tytul jest wszedzie wylaczony jawnie.

def fig1_liczebnosci():
    """Fig1: miesieczne liczebnosci naglowkow wedlug wydawcy."""
    from src import data as ds
    from src.visualization import plot_monthly_counts

    if config.RAW_CSV.exists():
        korpus = ds.load_and_prepare()
        return plot_monthly_counts(korpus, title=False)

    from src import figures_digitised as odczyt

    _ostrzez("Fig1", config.RAW_CSV)
    return plot_monthly_counts(odczyt.korpus_zastepczy(),
                               valid_start=odczyt.VALID_START,
                               test_start=odczyt.TEST_START, title=False)


def fig2_macierz_pomylek():
    """Fig2: macierz pomylek na tescie, udzial w wierszu klasy prawdziwej."""
    from src.visualization import plot_confusion_matrix

    y_true, y_pred = _etykiety_testu()
    return plot_confusion_matrix(y_true, y_pred, normalize="true", title=False)


def fig3_pewnosc():
    """Fig3: srednie prawdopodobienstwo klasy prawdziwej, wedlug wydawcy."""
    from src.evaluation import confidence_over_time, load_predictions
    from src.visualization import plot_metric_over_time

    if config.PROBS_CSV.exists():
        pewnosc = confidence_over_time(load_predictions())
    else:
        from src import figures_digitised as odczyt

        _ostrzez("Fig3", config.PROBS_CSV)
        pewnosc = odczyt.pewnosc_zastepcza()
    return plot_metric_over_time(pewnosc, "confidence", by_publisher=True,
                                 ylabel="Mean probability of true class",
                                 title=False)


def _etykiety_testu():
    """Etykiety prawdziwe i przewidywane dla zbioru testowego.

    Pierwszy wybor to plik predykcji, ten sam, z ktorego licza sie wszystkie
    tabele. Gdy go nie ma - a repozytorium go nie zawiera - etykiety odtwarzaja
    sie z zapisanej tabeli liczebnosci. Macierz pomylek zalezy wylacznie od
    dziewieciu liczb, wiec odtworzenie jest dokladne, nie przyblizone.
    """
    import numpy as np
    import pandas as pd

    if config.PROBS_CSV.exists():
        from src.evaluation import load_predictions

        test = load_predictions()
        test = test[test["split"] == "test"]
        return test["label"].to_numpy(), test["predicted_label"].to_numpy()

    tabela = config.OUTPUTS_DIR / "tabela_macierz_pomylek_test.csv"
    if not tabela.exists():
        raise FileNotFoundError(
            f"brak {config.PROBS_CSV.name} i {tabela.name}")
    tab = pd.read_csv(tabela, encoding="utf-8-sig").set_index("Wydawca prawdziwy")
    nazwy = [config.GAZETY[i] for i in sorted(config.GAZETY)]
    counts = tab.loc[nazwy, nazwy].to_numpy()
    y_true, y_pred = [], []
    for i, wiersz in enumerate(counts):
        for j, n in enumerate(wiersz):
            y_true.extend([i] * int(n))
            y_pred.extend([j] * int(n))
    return np.array(y_true), np.array(y_pred)


# Rysunki z artykulu, w numeracji z tekstu.
RYSUNKI_ARTYKULU = {
    "Fig1": fig1_liczebnosci,
    "Fig2": fig2_macierz_pomylek,
    "Fig3": fig3_pewnosc,
}


# --- rysunki spoza artykulu ------------------------------------------------
# Pozostale rysunki z outputs/figures/ nie maja numeru w tekscie, wiec zostaja
# przy nazwach opisowych. Sa tu na wypadek materialu uzupelniajacego.

def _profil_prawdopodobienstw():
    from src.evaluation import load_predictions, probability_profile_over_time
    from src.visualization import plot_probability_profile

    return plot_probability_profile(probability_profile_over_time(load_predictions()),
                                    title=False)


def _entropia():
    from src.evaluation import load_predictions, probability_profile_over_time
    from src.visualization import plot_metric_over_time

    profil = probability_profile_over_time(load_predictions())
    return plot_metric_over_time(profil, "entropy", ylabel="Mean entropy (bits)",
                                 events={"parliamentary election": "2023-10-15"},
                                 title=False)


def _macierz_liczebnosci():
    from src.visualization import plot_confusion_matrix

    y_true, y_pred = _etykiety_testu()
    return plot_confusion_matrix(y_true, y_pred, normalize=None, title=False)


RYSUNKI_DODATKOWE = {
    "AppFig_macierz_pomylek_test_liczebnosci": _macierz_liczebnosci,
    "AppFig_profil_prawdopodobienstw": _profil_prawdopodobienstw,
    "AppFig_entropia_w_czasie": _entropia,
}


def build(nazwy=None, out_dir: Path = PUBLICATION_DIR, formats=DEFAULT_FORMATS,
          dpi: int = DEFAULT_DPI, wszystkie: bool = False, verbose: bool = True):
    """Eksportuje rysunki do out_dir. Zwraca slownik nazwa -> lista sciezek.

    nazwy ogranicza eksport do wybranych pozycji; wszystkie=True dorzuca
    rysunki spoza artykulu. Brak danych dla pojedynczego rysunku nie przerywa
    eksportu pozostalych.
    """
    import matplotlib.pyplot as plt

    # Import przed ustawieniem profilu: src.visualization wola apply_style()
    # przy imporcie, wiec zaimportowany pozniej cofnalby profil na ekranowy.
    from src import visualization  # noqa: F401

    zrodla = dict(RYSUNKI_ARTYKULU)
    if wszystkie:
        zrodla.update(RYSUNKI_DODATKOWE)
    if nazwy:
        brakujace = set(nazwy) - set(zrodla)
        if brakujace:
            raise ValueError(f"Nieznane rysunki: {sorted(brakujace)}; "
                             f"dostepne: {sorted(zrodla)}")
        zrodla = {k: v for k, v in zrodla.items() if k in nazwy}

    # Profil drukarski ustawiamy raz, dla calego eksportu. Rysunki robocze
    # powstaja w osobnym procesie (notebooki), wiec nie ma czego przywracac.
    style.apply_style("print")

    wynik = {}
    for stem, funkcja in zrodla.items():
        try:
            fig = funkcja()
        except FileNotFoundError as blad:
            if verbose:
                print(f"  {stem}: pominiety, brak danych ({blad})")
            continue
        fig.tight_layout()
        wynik[stem] = zapisz(fig, stem, out_dir=out_dir, formats=formats, dpi=dpi)
        if verbose:
            print(f"  {stem}: {_wymiary(wynik[stem], dpi)} -> "
                  + ", ".join(p.name for p in wynik[stem]))
        plt.close(fig)
    return wynik


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Eksport rysunkow artykulu do osobnych plikow, "
                    "w szerokosci lamu i bez tytulow na obrazkach.")
    parser.add_argument("--all", action="store_true", dest="wszystkie",
                        help="eksportuj takze rysunki spoza artykulu")
    parser.add_argument("--formats", default=",".join(DEFAULT_FORMATS),
                        help="lista formatow po przecinku, np. pdf,tif,png "
                             f"(domyslnie {','.join(DEFAULT_FORMATS)})")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help=f"rozdzielczosc rastra (domyslnie {DEFAULT_DPI})")
    parser.add_argument("--out-dir", type=Path, default=PUBLICATION_DIR,
                        help=f"katalog docelowy (domyslnie {PUBLICATION_DIR})")
    parser.add_argument("--only", nargs="*", metavar="NAZWA",
                        help="eksportuj tylko wymienione rysunki, np. Fig2")
    args = parser.parse_args(argv)

    formats = tuple(f.strip().lstrip(".").lower()
                    for f in args.formats.split(",") if f.strip())
    print(f"Eksport do {args.out_dir} ({', '.join(formats)}, {args.dpi} dpi):")
    wynik = build(nazwy=args.only, out_dir=args.out_dir, formats=formats,
                  dpi=args.dpi, wszystkie=args.wszystkie)
    print(f"Zapisane rysunki: {len(wynik)}")
    if ODTWORZONE:
        print(f"UWAGA: {', '.join(ODTWORZONE)} powstaly z odczytu zapisanych PNG, "
              f"nie z danych zrodlowych. Po odzyskaniu data/ uruchom eksport "
              f"ponownie i podmien pliki.")
    return 0 if wynik else 1


if __name__ == "__main__":
    sys.exit(main())
