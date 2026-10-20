# -*- coding: utf-8 -*-
"""
Wczytywanie, czyszczenie i chronologiczny podzial korpusu naglowkow.
Logika odpowiada notebookowi 00 (EDA), opakowana w funkcje.
"""
import json

import pandas as pd

import config
from src.utils import clean_headline


def load_and_prepare(raw_csv=None):
    """Wczytuje surowy CSV, czysci naglowki, parsuje daty, filtruje od DATA_START
    i sortuje chronologicznie. Zwraca gotowy DataFrame z kolumna 'date_parsed'."""
    raw_csv = raw_csv or config.RAW_CSV
    data = pd.read_csv(raw_csv)

    # 1) czyszczenie naglowkow (usuniecie redakcyjnych dopiskow [..])
    before = data["text"].astype(str)
    data["text"] = before.map(clean_headline)
    empty = data["text"].str.len() == 0
    if empty.any():                       # zabezpieczenie: przywroc oryginal jesli pusty
        data.loc[empty, "text"] = before[empty]

    # 2) daty + filtr + sort
    data["date_parsed"] = pd.to_datetime(
        data["date"].str[:10], errors="coerce", dayfirst=True, format="mixed"
    )
    data = data.dropna(subset=["date_parsed", "text"])
    data = data[data["date_parsed"] >= pd.to_datetime(config.DATA_START)]
    data = data.sort_values("date_parsed").reset_index(drop=True)
    return data


def chronological_split(data):
    """Podzial po dacie: TEST = ostatni rok, VALID = przedostatni rok, TRAIN = reszta.
    Zwraca (train, valid, test, valid_start, test_start)."""
    max_date = data["date_parsed"].max()
    test_start = max_date - pd.DateOffset(years=config.TEST_YEARS)
    valid_start = max_date - pd.DateOffset(years=config.TEST_YEARS + config.VALID_YEARS)

    train = data[data["date_parsed"] < valid_start].copy()
    valid = data[(data["date_parsed"] >= valid_start) & (data["date_parsed"] < test_start)].copy()
    test = data[data["date_parsed"] >= test_start].copy()
    return train, valid, test, valid_start, test_start


def save_splits(train, valid, test, valid_start, test_start):
    """Zapisuje train/valid/test do CSV oraz metadane podzialu do split_info.json."""
    train.to_csv(config.TRAIN_CSV, index=False)
    valid.to_csv(config.VALID_CSV, index=False)
    test.to_csv(config.TEST_CSV, index=False)
    info = {
        "split_train_date": str(valid_start),   # granica TRAIN/VALID
        "split_val_date": str(test_start),       # granica VALID/TEST
        "train_size": len(train),
        "valid_size": len(valid),
        "test_size": len(test),
        "split_ratios": "TRAIN reszta / VALID 1 rok / TEST 1 rok",
    }
    with open(config.SPLIT_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


def class_distribution(data, gazety=None):
    """Slownik {nazwa_gazety: (liczba, udzial%)} dla podanego DataFrame."""
    gazety = gazety or config.GAZETY
    n = len(data)
    out = {}
    for label, name in gazety.items():
        c = int((data["label"] == label).sum())
        out[name] = (c, round(c / n * 100, 1) if n else 0.0)
    return out


def prepare_splits(raw_csv=None, lemmatize=False):
    """Etap 0 w calosci: wczytanie, czyszczenie, podzial chronologiczny i zapis.

    Sklada load_and_prepare + chronological_split + save_splits w jeden krok.
    Zwraca slownik z korpusem, podzbiorami, metadanymi podzialu i rozkladem klas,
    zeby notebook mogl pokazac dowolny z tych elementow.

    lemmatize=True dopisuje kolumne lemmatized_text PRZED podzialem, wiec
    podzbiory zapisuja sie raz i nie trzeba dzielic korpusu drugi raz.
    Lematyzacja trwa kilka minut.
    """
    data = load_and_prepare(raw_csv)
    if lemmatize:
        data = lemmatize_corpus(data)
    train, valid, test, valid_start, test_start = chronological_split(data)
    info = save_splits(train, valid, test, valid_start, test_start)
    return {
        "corpus": data,
        "train": train,
        "valid": valid,
        "test": test,
        "split_info": info,
        "class_distribution": class_distribution(data),
    }


def descriptive_stats(raw_csv=None, tokenizer=None, save=True):
    """Opis korpusu plus rozbicie ubytku rekordow przy czyszczeniu.

    Zwraca slownik z dwiema ramkami:

    per_publisher - liczebnosci calkowite i w podzbiorach, srednia liczba slow
    oraz srednia liczba tokenow BPE na naglowek, osobno dla kazdego wydawcy
    i dla calosci. Tokeny liczone sa bez znacznikow specjalnych, wiec podana
    wartosc opisuje sam naglowek, a nie dlugosc wejscia modelu (ta jest o dwa
    tokeny wieksza).

    cleaning - ile rekordow odpada na ktorym kroku oraz odsetek naglowkow
    dotknietych usunieciem dopiskow w nawiasach, osobno dla kazdego wydawcy.
    Odsetki liczone sa na korpusie SUROWYM, bo po filtrze dat czesc dopiskow
    juz z niego wypadla.

    Wymaga surowego korpusu (config.RAW_CSV) i zapisanych podzbiorow.
    """
    import pandas as pd

    from src.utils import clean_headline

    raw_csv = raw_csv or config.RAW_CSV
    gazety = config.GAZETY

    # --- rozbicie ubytku rekordow -------------------------------------------
    raw = pd.read_csv(raw_csv)
    przed = raw["text"].astype(str)
    po = przed.map(clean_headline)
    dotkniete = przed != po

    d = raw.copy()
    d["text"] = po
    d["date_parsed"] = pd.to_datetime(d["date"].str[:10], errors="coerce",
                                      dayfirst=True, format="mixed")
    bez_daty = int(d["date_parsed"].isna().sum())
    d = d.dropna(subset=["date_parsed", "text"])
    sprzed = int((d["date_parsed"] < pd.to_datetime(config.DATA_START)).sum())

    # Kazdy wiersz liczy odsetek wzgledem WEJSCIA tego kroku, nie wzgledem
    # korpusu surowego - inaczej odsetki krokow nastepujacych po sobie nie daja
    # sie porownac, a dla wydawcow mianownik jest inny niz dla calosci.
    wiersze = [{"Krok": "Bez daty publikacji", "Wejscie": len(raw), "Ubytek": bez_daty},
               {"Krok": f"Sprzed {config.DATA_START[:4]}",
                "Wejscie": len(raw) - bez_daty, "Ubytek": sprzed}]
    for label, nazwa in gazety.items():
        maska = raw["label"] == label
        wiersze.append({"Krok": f"Dopiski w nawiasach: {nazwa}",
                        "Wejscie": int(maska.sum()),
                        "Ubytek": int((dotkniete & maska).sum())})
    wiersze.append({"Krok": "Dopiski w nawiasach: razem", "Wejscie": len(raw),
                    "Ubytek": int(dotkniete.sum())})
    cleaning = pd.DataFrame(wiersze)
    cleaning["Udzial %"] = (100 * cleaning["Ubytek"] / cleaning["Wejscie"]).round(1)
    cleaning["Zostaje"] = cleaning["Wejscie"] - cleaning["Ubytek"]
    # Usuniecie dopiskow nie kasuje rekordow, wiec dla tych wierszy kolumna
    # "Zostaje" nie niesie sensu.
    cleaning.loc[cleaning["Krok"].str.startswith("Dopiski"), "Zostaje"] = pd.NA

    # --- opis korpusu --------------------------------------------------------
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.HERBERT_MODEL)

    podzbiory = load_splits()
    korpus = pd.concat(podzbiory.values(), ignore_index=True)
    teksty = korpus["text"].astype(str).tolist()
    bpe = pd.Series([len(x) for x in
                     tokenizer(teksty, add_special_tokens=False)["input_ids"]],
                    index=korpus.index)
    slowa = korpus["text"].astype(str).str.split().str.len()

    wiersze = []
    for label, nazwa in gazety.items():
        maska = korpus["label"] == label
        wiersz = {"Wydawca": nazwa, "Razem": int(maska.sum())}
        for klucz, ramka in podzbiory.items():
            wiersz[klucz] = int((ramka["label"] == label).sum())
        wiersz["Slowa"] = round(float(slowa[maska].mean()), 1)
        wiersz["Tokeny BPE"] = round(float(bpe[maska].mean()), 1)
        wiersze.append(wiersz)
    wiersz = {"Wydawca": "Razem", "Razem": len(korpus)}
    for klucz, ramka in podzbiory.items():
        wiersz[klucz] = len(ramka)
    wiersz["Slowa"] = round(float(slowa.mean()), 1)
    wiersz["Tokeny BPE"] = round(float(bpe.mean()), 1)
    wiersze.append(wiersz)
    per_publisher = pd.DataFrame(wiersze)

    if save:
        from src.evaluation import zapisz_tabele
        zapisz_tabele(per_publisher, "tabela_opis_korpusu")
        zapisz_tabele(cleaning, "tabela_ubytek_czyszczenie")

    return {"per_publisher": per_publisher, "cleaning": cleaning}


def lemmatize_corpus(data, column="text", extra_stopwords=None):
    """Dopisuje kolumne lemmatized_text (morfeusz2, bez stop slow).

    Lematyzacja kilkudziesieciu tysiecy naglowkow trwa kilka minut. Zbior stop
    slow to lista z pakietu stop_words uzupelniona o config.EXTRA_STOPWORDS,
    czyli formy czeste w naglowkach, ktore nie nosza tresci.

    Modyfikuje i zwraca kopie ramki.
    """
    import morfeusz2
    from stop_words import get_stop_words

    from src.utils import lemmatize_headline

    stop = set(get_stop_words("pl")) | set(config.EXTRA_STOPWORDS)
    if extra_stopwords:
        stop |= set(extra_stopwords)
    morf = morfeusz2.Morfeusz()

    data = data.copy()
    lematy = [lemmatize_headline(str(t), morf, stop) for t in data[column]]
    data["lemmatized_text"] = [" ".join(l) for l in lematy]
    return data


def top_ngrams(texts, n=1, top=30):
    """Najczestsze n-gramy w korpusie. Zwraca DataFrame [ngram, liczba].

    texts - teksty po lematyzacji. Dla n=1 daje rozklad czestosci wyrazow.
    """
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer

    wektoryzator = CountVectorizer(ngram_range=(n, n), max_features=top)
    X = wektoryzator.fit_transform(texts)
    liczby = X.sum(axis=0).A1
    tab = pd.DataFrame({"ngram": wektoryzator.get_feature_names_out(), "liczba": liczby})
    return tab.sort_values("liczba", ascending=False).reset_index(drop=True)


def tfidf_value_stats(texts, max_features=1000, ngram_max=3):
    """Rozklad niezerowych wag TF-IDF w korpusie: statystyki i gorne kwantyle.

    Sluzy do ustawienia progu odciecia rzadkich termow - pokazuje, od ktorego
    kwantyla wagi przestaja rosnac liniowo. Zwraca Series.
    """
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(max_features=max_features,
                        ngram_range=(1, ngram_max)).fit_transform(texts)
    wagi = X.data
    opis = {"min": wagi.min(), "max": wagi.max(),
            "srednia": wagi.mean(), "mediana": float(np.median(wagi))}
    for q in range(90, 100):
        opis[f"kwantyl {q}%"] = float(np.quantile(wagi, q / 100))
    return pd.Series(opis)


def load_splits(as_int_labels=True):
    """Wczytuje zapisane podzbiory train/valid/test. Zwraca slownik DataFrame.

    Podzbiory powstaja w prepare_splits (notebook 00). Gdy ktoregos brakuje,
    zglasza blad zamiast liczyc podzial po raz drugi - podzial ma jedno zrodlo.
    """
    import pandas as pd

    sciezki = {"train": config.TRAIN_CSV, "valid": config.VALID_CSV, "test": config.TEST_CSV}
    brak = [str(s) for s in sciezki.values() if not s.exists()]
    if brak:
        raise FileNotFoundError(
            f"Brak podzbiorow: {brak}. Uruchom najpierw notebook 00 "
            f"(data.prepare_splits).")

    out = {}
    for nazwa, sciezka in sciezki.items():
        d = pd.read_csv(sciezka)
        if as_int_labels:
            d["label"] = d["label"].astype(int)
        out[nazwa] = d
    return out
