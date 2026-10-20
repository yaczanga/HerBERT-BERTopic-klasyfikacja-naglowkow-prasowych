# -*- coding: utf-8 -*-
"""
Rysunki. Jedyne zrodlo barw i proporcji to src.plot_style.

Napisy na rysunkach sa po angielsku i z kropka dziesietna, odmiennie od jezyka
README. Jest to swiadome odstepstwo od domyslnego rejestru polskiego systemu
wizualnego, opisane w README w sekcji o rysunkach.
"""
from __future__ import annotations

import numpy as np

import config
from src import plot_style as style

style.apply_style()


def _zapisz(fig, save_path):
    """Zapisuje rysunek pod podana sciezka i zwraca go dalej.

    Rozdzielczosc, kadrowanie i barwa tla pochodza z systemu wizualnego, zeby
    wszystkie rysunki w outputs/figures/ mialy te same parametry zapisu.
    """
    if save_path:
        fig.savefig(save_path, dpi=style.SAVE_DPI, bbox_inches='tight',
                    facecolor=style.BACKGROUND)
    return fig

# Skroty gazet uzywane w naglowkach macierzy - kolejnosc z config.GAZETY.
PUBLISHER_LABELS = ['Niezależna', 'OKO.press', 'RP']


def plot_confusion_matrix(y_true, y_pred, labels=None, title=None,
                          normalize='true', save_path=None):
    """Macierz pomylek klasyfikatora wydawcy.

    Rola koloru jest SEKWENCYJNA - komorka koduje wielkosc (udzial naglowkow),
    wiec rampa ma jeden odcien od bieli. Macierz nie jest symetryczna (wiersz to
    klasa prawdziwa, kolumna to predykcja), wiec przekatnej nie maskujemy.

    normalize='true' podpisuje komorki udzialem w wierszu (czulosc na przekatnej)
    oraz liczba bezwzgledna; normalize=None pokazuje same liczebnosci.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    labels = labels or PUBLISHER_LABELS
    idx = list(range(len(labels)))
    counts = confusion_matrix(y_true, y_pred, labels=idx)

    if normalize == 'true':
        shown = counts / counts.sum(axis=1, keepdims=True)
        cbar_label = 'Share of true-class headlines'
    else:
        shown = counts.astype(float)
        cbar_label = 'Headlines'

    fig, ax = plt.subplots(figsize=style.SQUARE)
    # Siatka z rcParams jest dla slupkow - na macierzy przeszkadza.
    ax.grid(False)
    im = ax.imshow(shown, cmap=style.SEQUENTIAL_CMAP, vmin=0,
                   vmax=shown.max())

    # Obramowanie komorek: jasne wypelnienia na bialym tle inaczej zlewaja sie
    # z tlem figury.
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which='minor', color=style.GRID, linewidth=1.0)
    ax.tick_params(which='minor', length=0)

    # Prog przelaczenia barwy podpisu - na ciemnym koncu rampy czarny tekst ginie.
    threshold = shown.max() * 0.55
    for i in idx:
        for j in idx:
            dark = shown[i, j] > threshold
            colour = style.BACKGROUND if dark else style.TEXT
            if normalize == 'true':
                primary = f'{shown[i, j] * 100:.1f}%'
                secondary = f'{counts[i, j]:,}'
            else:
                primary = f'{counts[i, j]:,}'
                secondary = ''
            ax.text(j, i - (0.10 if secondary else 0.0), primary,
                    ha='center', va='center', color=colour, fontsize=12)
            if secondary:
                ax.text(j, i + 0.17, secondary, ha='center', va='center',
                        color=colour, fontsize=9.5, alpha=0.85)

    ax.set_xticks(idx, labels)
    ax.set_yticks(idx, labels)
    ax.set_xlabel('Predicted publisher')
    ax.set_ylabel('True publisher')
    ax.set_title(title or 'Confusion matrix, test set')
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.set_label(cbar_label, color=style.TEXT_SECONDARY)
    cbar.ax.tick_params(colors=style.TEXT_SECONDARY, length=0)

    return _zapisz(fig, save_path)

def plot_monthly_counts(data, valid_start=None, test_start=None, title=None,
                        save_path=None):
    """Miesieczna liczba naglowkow na wydawce, z zaznaczonym podzialem korpusu.

    Rola koloru jest TOZSAMOSCIOWA - seria to wydawca - wiec barwy pochodza
    z style.PUBLISHER_COLORS, te same co na pozostalych rysunkach projektu.
    Granice podzialu i pas testowy sa tuszem nie-danych, wiec nosza barwy
    neutralne i nie wchodza do legendy.

    valid_start i test_start pochodza z data.chronological_split; gdy sa None,
    czytane sa ze split_info.json.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.utils import load_split_dates

    if valid_start is None or test_start is None:
        valid_start, test_start = load_split_dates()
    valid_start, test_start = pd.to_datetime(valid_start), pd.to_datetime(test_start)

    fig, ax = plt.subplots(figsize=style.WIDE)
    for label in sorted(config.GAZETY):
        miesieczne = (data[data['label'] == label]
                      .set_index('date_parsed').resample('MS').size())
        ax.plot(miesieczne.index, miesieczne.values, label=PUBLISHER_LABELS[label],
                marker='o', markersize=3, linewidth=1.8,
                **style.PUBLISHER_STYLES[label])

    poczatek, koniec = data['date_parsed'].min(), data['date_parsed'].max()
    ax.set_xlim(poczatek, koniec)
    gora = ax.get_ylim()[1]

    # Pas testowy zacieniony tak samo jak na rysunku bledow w czasie, zeby oba
    # rysunki czytalo sie tym samym kluczem.
    ax.axvspan(test_start, koniec, color=style.MUTED, alpha=0.10, zorder=0)
    for granica in (valid_start, test_start):
        ax.axvline(granica, color=style.AXES, linestyle='--', linewidth=1.2, zorder=1)
    for x, tekst, wyrownanie in ((valid_start, 'train  ', 'right'),
                                 (valid_start, '  valid', 'left'),
                                 (test_start, '  test', 'left')):
        ax.text(x, gora * 0.98, tekst, ha=wyrownanie, va='top',
                fontsize=9, color=style.TEXT_SECONDARY)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_ylabel('Headlines per month')
    ax.set_title(title or 'Monthly headline counts by publisher')
    ax.legend(loc='upper left', ncol=3)
    fig.tight_layout()

    return _zapisz(fig, save_path)


def plot_topic_error_series(soft, topic_model=None, topics=None, n_cols=2,
                            value='misclassified_count', ylabel=None,
                            max_topics=None, split_date=None, title=None,
                            save_path=None):
    """Oczekiwana liczba bledow w miesiacu: panel na temat, seria na wydawce.

    soft - ramka z evaluation.soft_error_stats (kolumny date_parsed, label,
    topic oraz kolumna value). topic_model sluzy tylko do podpisania paneli
    czolowymi slowami tematu. max_topics ogranicza rysunek do najwiekszych
    tematow, liczonych suma kolumny value.

    Zakres testowy jest zacieniony; gdy split_date nie podana, data granicy
    VALID/TEST czytana jest ze split_info.json.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.utils import load_split_dates, topic_label

    d = soft.copy()
    d['date_parsed'] = pd.to_datetime(d['date_parsed'])

    if topics is None:
        topics = sorted(d['topic'].unique())
    if max_topics:
        czolowe = (d[d['topic'].isin(topics)].groupby('topic')[value].sum()
                   .sort_values(ascending=False).head(max_topics).index)
        topics = [t for t in topics if t in set(czolowe)]
    if not topics:
        raise ValueError('Brak tematow do narysowania.')

    # Pelna os miesieczna - bez niej miesiac bez bledow znika, a linia
    # przeskakuje luke, sugerujac ciaglosc, ktorej w danych nie ma.
    miesiace = pd.date_range(d['date_parsed'].min(), d['date_parsed'].max(), freq='MS')
    if split_date is None:
        _, split_date = load_split_dates()

    n_rows = -(-len(topics) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, squeeze=False,
                             figsize=style.panel_grid(n_cols, n_rows))
    osie = axes.flatten()

    for i, (ax, temat) in enumerate(zip(osie, topics)):
        for label in sorted(config.GAZETY):
            seria = (d[(d['topic'] == temat) & (d['label'] == label)]
                     .groupby('date_parsed')[value].sum()
                     .reindex(miesiace, fill_value=0.0))
            ax.plot(miesiace, seria.values, label=PUBLISHER_LABELS[label],
                    marker='o', markersize=3, linewidth=1.8,
                    **style.PUBLISHER_STYLES[label])

        nazwa = topic_label(topic_model, temat, fallback='Topic {}') if topic_model else None
        ax.set_title(f'Topic {temat + 1}' + (f': {nazwa}' if nazwa else ''), fontsize=11)

        # Zakres testowy to tusz nie-danych, wiec nosi MUTED i nie wchodzi do legendy.
        if split_date is not None and split_date <= miesiace[-1]:
            ax.axvspan(split_date, miesiace[-1], color=style.MUTED, alpha=0.10, zorder=0)
            if i == 0:
                ax.annotate('test set', xy=(split_date, 1.0), xycoords=('data', 'axes fraction'),
                            xytext=(4, -10), textcoords='offset points',
                            fontsize=8, color=style.TEXT_SECONDARY)

        if i % n_cols == 0:
            ax.set_ylabel(ylabel or 'Expected errors per month')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', labelrotation=45)

    for ax in osie[len(topics):]:
        fig.delaxes(ax)

    uchwyty, etykiety = osie[0].get_legend_handles_labels()
    fig.legend(uchwyty, etykiety, loc='lower center', ncol=len(etykiety),
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title or 'Expected classification errors by topic and publisher',
                 x=0.0, ha='left', fontsize=13, color=style.TEXT)
    fig.tight_layout()

    return _zapisz(fig, save_path)


# Predykcje wczytuje i macierze rysuje evaluation.build_article_tables, wolane
# z notebooka 04. Wczesniej byla tu funkcja opakowujaca te dwa kroki, ale nic
# jej nie wolalo.


def _mark_split_and_events(ax, end, split_date=None, events=None):
    """Nanosi granice VALID/TEST i daty kontekstowe - tusz nie-danych.

    Linie nosza barwy neutralne i nie trafiaja do legendy: oznaczaja miejsce na
    osi czasu, nie kategorie w danych.

    end podajemy jawnie jako date. Wziecie go z ax.get_xlim() daje liczbe
    zmiennoprzecinkowa, ktora konwerter dat pandas czyta jako mikrosekundy od
    1970 roku - pas zaznaczenia rysuje sie wtedy wstecz, przez caly wykres.
    """
    import pandas as pd

    if split_date is not None:
        split_date = pd.to_datetime(split_date)
        ax.axvspan(split_date, pd.to_datetime(end), color=style.MUTED, alpha=0.10, zorder=0)
        ax.annotate('test set', xy=(split_date, 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(4, -10), textcoords='offset points',
                    fontsize=8, color=style.TEXT_SECONDARY)
    for opis, data_zdarzenia in (events or {}).items():
        data_zdarzenia = pd.to_datetime(data_zdarzenia)
        ax.axvline(data_zdarzenia, color=style.AXES, linestyle=':', linewidth=1.2, zorder=1)
        ax.annotate(opis, xy=(data_zdarzenia, 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(4, -24), textcoords='offset points',
                    fontsize=8, color=style.TEXT_SECONDARY)


def plot_metric_over_time(frame, value, by_publisher=False, ylabel=None, title=None,
                          split_date=None, events=None, save_path=None):
    """Szereg miesieczny jednej miary, lacznie albo w podziale na wydawcow.

    by_publisher=True rysuje trzy serie w barwach PUBLISHER_COLORS (rola
    tozsamosciowa, te same barwy co na pozostalych rysunkach). Przy jednej
    serii legendy nie ma - nazywa ja tytul.

    frame - ramka z kolumna date oraz kolumna o nazwie value; przy podziale na
    wydawcow dodatkowo z kolumna label.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.utils import load_split_dates

    if split_date is None:
        _, split_date = load_split_dates()

    d = frame.copy()
    d['date'] = pd.to_datetime(d['date'])

    fig, ax = plt.subplots(figsize=style.WIDE)
    if by_publisher:
        for label in sorted(config.GAZETY):
            seria = d[d['label'] == label].sort_values('date')
            ax.plot(seria['date'], seria[value], label=PUBLISHER_LABELS[label],
                    marker='o', markersize=3, linewidth=1.8,
                    **style.PUBLISHER_STYLES[label])
        ax.legend(loc='lower left', ncol=3)
    else:
        seria = d.sort_values('date')
        ax.plot(seria['date'], seria[value], color=style.REFERENCE,
                marker='o', markersize=3, linewidth=1.8)

    ax.set_xlim(d['date'].min(), d['date'].max())
    _mark_split_and_events(ax, d['date'].max(), split_date, events)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_ylabel(ylabel or value)
    # Awaryjny tytul nie moze byc sama nazwa kolumny - to identyfikator z kodu,
    # nie opis. Notebooki podaja title jawnie, ten wariant jest zabezpieczeniem.
    ax.set_title(title or f'{value} over time')
    fig.tight_layout()

    return _zapisz(fig, save_path)


def plot_probability_profile(profile, title=None, split_date=None, save_path=None):
    """Sredni rozklad prawdopodobienstwa predykcji w czasie, warstwami.

    Warstwy sumuja sie do jednosci, wiec rysunek pokazuje SKLAD przewidywania
    w miesiacu, a nie poziom. Rola koloru jest tozsamosciowa - warstwa to
    wydawca - wiec barwy pochodza z PUBLISHER_COLORS, a desen z PUBLISHER_HATCHES.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.utils import load_split_dates

    if split_date is None:
        _, split_date = load_split_dates()

    d = profile.copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.sort_values('date')
    etykiety = sorted(config.GAZETY)

    fig, ax = plt.subplots(figsize=style.WIDE)
    warstwy = ax.stackplot(d['date'],
                           *[d[f'probability_topic_{label}'] for label in etykiety],
                           labels=[PUBLISHER_LABELS[label] for label in etykiety],
                           colors=[style.PUBLISHER_COLORS[label] for label in etykiety],
                           alpha=0.85)
    # Stackplot nie ma kroju linii, a dwie z trzech barw wydawcow maja niemal
    # te sama jasnosc - rozroznia je desen, patrz PUBLISHER_HATCHES.
    for warstwa, label in zip(warstwy, etykiety):
        warstwa.set_hatch(style.PUBLISHER_HATCHES[label])
        warstwa.set_edgecolor(style.BACKGROUND)
        warstwa.set_linewidth(0.0)
    ax.set_xlim(d['date'].min(), d['date'].max())
    ax.set_ylim(0, 1)
    if split_date is not None:
        ax.axvline(pd.to_datetime(split_date), color=style.TEXT, linestyle='--', linewidth=1.4)
        # Podpis lezy na deseniu gornej warstwy, wiec dostaje tlo - bez niego
        # kropki przebijaja przez litery.
        ax.annotate('test set', xy=(pd.to_datetime(split_date), 1.0),
                    xycoords=('data', 'axes fraction'), xytext=(4, -10),
                    textcoords='offset points', fontsize=8, color=style.TEXT,
                    bbox=dict(facecolor=style.BACKGROUND, edgecolor='none',
                              boxstyle='square,pad=0.2'))
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_ylabel('Mean predicted probability')
    ax.set_title(title or 'Mean predicted publisher profile over time')
    ax.legend(loc='lower left', ncol=3)
    fig.tight_layout()

    return _zapisz(fig, save_path)


def plot_merge_levels(results, x='n_topics', highlight=None, title=None, save_path=None):
    """Spojnosc tematow na kolejnych poziomach scalania hierarchicznego.

    Cztery panele: trzy miary spojnosci oraz ich suma po przeskalowaniu min-max,
    czyli kryterium wyboru poziomu. W kazdym panelu jest jedna seria, wiec
    legendy nie ma - nazywa ja tytul panelu.

    highlight wskazuje wiersz wybranego poziomu; punkt dostaje barwe ACCENT,
    bo jest jednym wyroznionym elementem na tle pozostalych.
    """
    import matplotlib.pyplot as plt

    miary = [('umass', 'UMass'), ('uci', 'UCI'), ('npmi', 'NPMI'),
             ('sum_normalised', 'Sum of min-max scaled measures')]
    d = results.sort_values(x)

    fig, axes = plt.subplots(1, 4, figsize=style.panel_grid(4, 1))
    for ax, (kolumna, opis) in zip(axes, miary):
        if kolumna not in d.columns:
            continue
        ax.plot(d[x], d[kolumna], color=style.REFERENCE, marker='o',
                markersize=3, linewidth=1.8)
        if highlight is not None and highlight in results.index:
            ax.plot(results.loc[highlight, x], results.loc[highlight, kolumna],
                    marker='o', markersize=9, color=style.ACCENT, linestyle='none',
                    zorder=5)
        ax.set_title(opis, fontsize=11)
        ax.set_xlabel('Topics after merging' if x == 'n_topics' else 'Merge distance')

    fig.suptitle(title or 'Topic coherence across merge levels',
                 x=0.0, ha='left', fontsize=13, color=style.TEXT)
    fig.tight_layout()

    return _zapisz(fig, save_path)
