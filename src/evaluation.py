# -*- coding: utf-8 -*-
"""
Ewaluacja i wizualizacja wynikow klasyfikatora (tabele metryk, macierze pomylek).
Logika odpowiada notebookom 01a / 03.
"""
import numpy as np
import pandas as pd

import config
from src.utils import rel


def metrics_table(y_true, y_pred, split_name, gazety=None):
    """Tabela metryk per gazeta + srednia wazona (jak w notebooku 01a)."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    gazety = gazety or config.GAZETY
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    acc = accuracy_score(y_true, y_pred)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    rows = []
    for idx, name in gazety.items():
        rows.append({"Gazeta": name, "Zbiór": split_name,
                     "Accuracy": f"{acc:.4f}" if idx == 0 else "",
                     "Precision": f"{precision[idx]:.4f}", "Recall": f"{recall[idx]:.4f}",
                     "F1-Score": f"{f1[idx]:.4f}", "Liczba": int(support[idx])})
    rows.append({"Gazeta": "Średnia ważona", "Zbiór": split_name, "Accuracy": f"{acc:.4f}",
                 "Precision": f"{p_w:.4f}", "Recall": f"{r_w:.4f}", "F1-Score": f"{f1_w:.4f}",
                 "Liczba": int(sum(support))})
    return pd.DataFrame(rows)


def load_predictions(probs_csv=None):
    """Predykcje finalnego klasyfikatora z kolumnami label/predicted_label/split.

    Kolumne date_parsed zwraca jako date, zeby kazdy notebook czytal ja tak
    samo; wiersze bez daty zostaja, bo tabele metryk ich nie potrzebuja.
    """
    probs_csv = probs_csv or config.PROBS_CSV
    data = pd.read_csv(probs_csv)
    brakujace = {"label", "predicted_label", "split"} - set(data.columns)
    if brakujace:
        raise ValueError(f"W {probs_csv} brakuje kolumn: {sorted(brakujace)}")
    if "date_parsed" in data.columns:
        data["date_parsed"] = pd.to_datetime(data["date_parsed"], errors="coerce")
    elif "date" in data.columns:
        data["date_parsed"] = pd.to_datetime(data["date"].str[:10], errors="coerce",
                                             dayfirst=True, format="mixed")
    return data


def baselines_table(data, skip_baselines=False, include_frozen=True):
    """Klasyfikator dostrojony na tle punktow odniesienia, accuracy i F1-macro.

    Uklad wierszy tabeli punktow odniesienia: klasa wiekszosciowa, TF-IDF z regresja
    logistyczna, TF-IDF z liniowym SVM, HerBERT bez dostrojenia, HerBERT
    dostrojony.

    data - ramka z load_predictions(). Zwraca (DataFrame, slownik ze slownikiem
    TF-IDF albo None, gdy punktow odniesienia nie liczono).
    """
    from sklearn.metrics import accuracy_score, f1_score
    from src.classifier import baselines

    rows, vocab = [], None
    if not skip_baselines:
        base, vocab = baselines(
            data[data["split"] == "train"],
            data[data["split"] == "test"],
            valid_df=data[data["split"] == "valid"],
            return_vocab=True,
            include_frozen=include_frozen,
        )
        rows.append(base)

    herbert = {"model": "HerBERT (dostrojony)"}
    for split in ("valid", "test"):
        d = data[data["split"] == split]
        herbert[f"accuracy_{split}"] = accuracy_score(d["label"], d["predicted_label"])
        herbert[f"f1_macro_{split}"] = f1_score(d["label"], d["predicted_label"], average="macro")
    rows.append(pd.DataFrame([herbert]))

    out = pd.concat(rows, ignore_index=True)
    return out[["model", "accuracy_valid", "f1_macro_valid",
                "accuracy_test", "f1_macro_test"]], vocab


def error_topics_table(csv_path=None, topics_path=None, gazety=None):
    """Tematy bledow klasyfikatora z kierunkami pomylek.

    Czyta przypisania tematow z misclassified_with_topics.csv i reprezentacje
    z zapisanego modelu, wiec nie wymaga GPU ani liczenia embeddingow od nowa.
    Zwraca DataFrame albo None, gdy ktoregos z plikow brakuje.
    """
    import json

    gazety = gazety or config.GAZETY
    skrot = {0: "N", 1: "O", 2: "RP"}
    csv_path = csv_path or (config.DATA / "misclassified_with_topics.csv")
    topics_path = topics_path or (config.MISCLASSIFIED_DIR / "topics.json")
    if not csv_path.exists() or not topics_path.exists():
        return None

    mis = pd.read_csv(csv_path)
    reps = json.loads(topics_path.read_text(encoding="utf-8"))["topic_representations"]

    rows = []
    for t in sorted(mis["mis_topic"].unique()):
        d = mis[mis["mis_topic"] == t]
        rep = reps.get(str(t)) or reps.get(t) or []
        prawdziwy = d["label"].value_counts(normalize=True)
        przewidziany = d["predicted_label"].value_counts(normalize=True)
        para = d.groupby(["label", "predicted_label"]).size().sort_values(ascending=False)
        (lab, pred), ile = para.index[0], para.iloc[0]
        rows.append({
            "Topic": t if t != -1 else "-1 (outliers)",
            "Top terms": ", ".join(w for w, _ in rep[:6] if w),
            "N": len(d),
            "True publisher (N / O / RP, %)":
                " / ".join(f"{prawdziwy.get(i, 0) * 100:.0f}" for i in (0, 1, 2)),
            "Predicted publisher (N / O / RP, %)":
                " / ".join(f"{przewidziany.get(i, 0) * 100:.0f}" for i in (0, 1, 2)),
            "Dominant confusion":
                f"{skrot[lab]}→{skrot[pred]} ({ile / len(d) * 100:.0f}%)",
        })

    tab = pd.DataFrame(rows)
    realne = tab[tab["Topic"] != "-1 (outliers)"].sort_values("N", ascending=False)
    return pd.concat([realne, tab[tab["Topic"] == "-1 (outliers)"]], ignore_index=True)


def zapisz_tabele(tab, nazwa, markdown=False):
    """Zapisuje tabele do outputs/ jako CSV, opcjonalnie takze jako Markdown.

    Kolumny liczbowe zaokraglane sa do trzech miejsc: tabela jest artefaktem do
    czytania, a kolumna o roznej liczbie miejsc po przecinku nie da sie porownac
    wzrokiem. Pelna precyzja zostaje w pliku predykcji.
    """
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    tab = tab.copy()
    for kolumna in tab.select_dtypes(include="float").columns:
        tab[kolumna] = tab[kolumna].round(3)
    tab.to_csv(config.OUTPUTS_DIR / f"{nazwa}.csv", index=False, encoding="utf-8-sig")
    if markdown:
        kol = list(tab.columns)
        linie = ["| " + " | ".join(kol) + " |", "|" + "---|" * len(kol)]
        linie += ["| " + " | ".join(str(r[c]) for c in kol) + " |" for _, r in tab.iterrows()]
        (config.OUTPUTS_DIR / f"{nazwa}.md").write_text("\n".join(linie), encoding="utf-8")


# Macierze pomylek rysuje src/visualization.py (plot_confusion_matrix), ktory
# korzysta z systemu wizualnego projektu z src/plot_style.py. Poprzednia wersja
# oparta na ConfusionMatrixDisplay zostala stad usunieta, zeby nie bylo dwoch
# sposobow rysowania tego samego obiektu.


def build_article_tables(skip_baselines=False, save=True, include_frozen=True):
    """Etap 4 w calosci: wszystkie tabele i rysunki z JEDNEGO pliku predykcji.

    Dzieki temu, ze zrodlem jest jeden plik, tabela punktow odniesienia, tabela metryk
    per wydawca, liczba bledow i macierze pomylek pochodza z tego samego
    przebiegu. Liczby podane w README biora sie stad.

    skip_baselines=True pomija uczenie punktow odniesienia.
    include_frozen=False pomija sam baseline na zamrozonym enkoderze; pierwsze
    jego policzenie wymaga przejscia korpusu przez HerBERTa i trwa kilka minut,
    kolejne czytaja embeddingi z dysku.
    save=False nie zapisuje nic na dysk.

    Zwraca slownik: tabele (DataFrame), slownik TF-IDF, liczbe bledow na tescie
    i sciezki zapisanych rysunkow.
    """
    from src import visualization

    data = load_predictions()
    t4, vocab = baselines_table(data, skip_baselines, include_frozen=include_frozen)
    t5 = pd.concat([metrics_table(d["label"], d["predicted_label"], nazwa)
                    for nazwa, d in (("walidacyjny", data[data["split"] == "valid"]),
                                     ("testowy", data[data["split"] == "test"]))],
                   ignore_index=True)
    tc = error_topics_table()

    test = data[data["split"] == "test"]
    valid = data[data["split"] == "valid"]
    n_err = int((test["label"] != test["predicted_label"]).sum())

    rysunki = []
    for subset, normalize, tytul, nazwa in (
            (test, "true", "Confusion matrix, test set", "01_macierz_pomylek_test_udzialy.png"),
            (test, None, "Confusion matrix, test set", "02_macierz_pomylek_test_liczebnosci.png"),
            (valid, "true", "Confusion matrix, validation set",
             "03_macierz_pomylek_walidacja_udzialy.png")):
        sciezka = config.FIGURES / nazwa if save else None
        visualization.plot_confusion_matrix(
            subset["label"], subset["predicted_label"], normalize=normalize,
            title=tytul, save_path=sciezka)
        if sciezka:
            rysunki.append(sciezka)

    if save:
        zapisz_tabele(t4, "tabela_punkty_odniesienia")
        zapisz_tabele(t5, "tabela_metryki_wydawcy")
        if tc is not None:
            zapisz_tabele(tc, "tabela_tematy_bledow", markdown=True)

    return {
        "baselines": t4,
        "per_publisher": t5,
        "error_topics": tc,
        "tfidf_vocab": vocab,
        "n_errors_test": n_err,
        "accuracy_test": 1 - n_err / len(test),
        "figures": rysunki,
    }


def hard_error_stats(data, topic_model=None, min_articles=10, save=True, out_csv=None):
    """Twarde pomylki w miesiacu, w podziale na wydawce prawdziwego i temat.

    Twarda pomylka to zdarzenie zero-jedynkowe: label != predicted_label.
    Grupy ponizej min_articles naglowkow sa odrzucane, bo udzial policzony
    na kilku obserwacjach nie nosi informacji. Naglowki odstajace (topic == -1)
    sa pomijane - nie naleza do zadnego tematu.

    topic_model dolacza nazwe i reprezentacje tematu; bez niego tabela ma same
    numery. Zwraca DataFrame.
    """
    d = data[data["topic"] != -1].copy()
    d["year"] = d["date_parsed"].dt.year
    d["month"] = d["date_parsed"].dt.month
    d["is_misclassified"] = d["label"] != d["predicted_label"]

    stat = d.groupby(["label", "topic", "year", "month"]).agg(
        total_articles=("is_misclassified", "size"),
        misclassified_count=("is_misclassified", "sum"),
    ).reset_index()
    stat["misclassified_percent"] = stat["misclassified_count"] / stat["total_articles"] * 100

    if topic_model is not None:
        info = topic_model.get_topic_info()[["Topic", "Name", "Representation"]]
        stat = stat.merge(info, left_on="topic", right_on="Topic", how="left").drop(columns=["Topic"])

    stat = stat[stat["total_articles"] >= min_articles]
    stat = stat.sort_values(["misclassified_percent", "misclassified_count"], ascending=False)

    if save:
        out_csv = out_csv or (config.DATA / "dane_zle_sklasyfikowane_statystyki.csv")
        stat.to_csv(out_csv, index=False)
        print(f"Zapisano {out_csv.name}: {len(stat)} wierszy")
    return stat


def soft_error_stats(data, save=True, out_csv=None):
    """Oczekiwana liczba bledow w miesiacu, w podziale na wydawce i temat.

    Twarda pomylka jest zdarzeniem zero-jedynkowym, wiec szereg miesieczny
    pojedynczego tematu jest rzadki i skacze miedzy zerem a kilkoma bledami.
    Zamiast niej liczymy wartosc oczekiwana: wklad naglowka to
    P(blad) = 1 - P(klasa prawdziwa), czyli prawdopodobienstwo, ktore
    klasyfikator rozlozyl na pozostalych wydawcow. Suma po naglowkach miesiaca
    daje oczekiwana liczbe bledow, ktora zmienia sie plynnie.

    Naglowki odstajace (topic == -1) sa pomijane. Zwraca DataFrame.
    """
    kolumny_p = [f"probability_topic_{i}" for i in sorted(config.GAZETY)]
    d = data[data["topic"] != -1].dropna(subset=kolumny_p + ["date_parsed"]).copy()

    probs = d[kolumny_p].to_numpy(dtype=float)
    etykiety = d["label"].to_numpy(dtype=int)
    d["p_mis"] = 1.0 - probs[np.arange(len(d)), etykiety]
    d["month_str"] = d["date_parsed"].dt.to_period("M")

    stat = d.groupby(["month_str", "label", "topic"]).agg(
        total_articles=("p_mis", "size"),
        misclassified_count=("p_mis", "sum"),
    ).reset_index()

    stat["date_parsed"] = stat["month_str"].dt.to_timestamp()
    stat["year"] = stat["date_parsed"].dt.year
    stat["month"] = stat["date_parsed"].dt.month
    stat["month_str"] = stat["month_str"].astype(str)
    stat["misclassified_percent"] = stat["misclassified_count"] / stat["total_articles"] * 100
    stat = stat[["month_str", "label", "topic", "total_articles", "misclassified_count",
                 "misclassified_percent", "date_parsed", "year", "month"]]

    if save:
        out_csv = out_csv or (config.DATA / "dane_wygladzone_miekkie.csv")
        stat.to_csv(out_csv, index=False)
        print(f"Zapisano {out_csv.name}: {len(stat)} wierszy")
    return stat


def build_error_time_series(probs_csv=None, cache_csv=None, device="cuda",
                            force_transform=False, min_articles=10, save=True):
    """Etap 2 w calosci: tematy dla korpusu + statystyki bledow w czasie.

    Sklada wczytanie predykcji, wczytanie zapisanego modelu BERTopic,
    przypisanie tematow (z cache na dysku) oraz obie tabele bledow - twardych
    i oczekiwanych. Pierwszy przebieg wymaga GPU i trwa kilka minut; kolejne
    biora tematy z cache.

    Zwraca slownik: korpus z kolumna topic, obie tabele, model i jego sciezke.
    """
    from src import topics as tpx

    probs_csv = probs_csv or config.PROBS_CSV
    data = pd.read_csv(probs_csv)
    if "date_parsed" in data.columns:
        data["date_parsed"] = pd.to_datetime(data["date_parsed"], errors="coerce")
    else:
        data["date_parsed"] = pd.to_datetime(data["date"].str[:10], errors="coerce",
                                             dayfirst=True, format="mixed")
    data = data.dropna(subset=["date_parsed", "text"])
    data = data[data["date_parsed"] >= pd.to_datetime(config.DATA_START)].reset_index(drop=True)
    print(f"Korpus do analizy czasowej: {len(data)} naglowkow "
          f"({data['date_parsed'].min().date()} - {data['date_parsed'].max().date()})")

    topic_model, sciezka = tpx.load_saved_model(device=device)
    print(f"Model BERTopic: {rel(sciezka)}")

    tematy, _ = tpx.assign_topics(data["text"].astype(str).tolist(), topic_model,
                                  cache_csv=cache_csv, force=force_transform)
    data["topic"] = tematy
    odstajace = float((data["topic"] == -1).mean())
    print(f"Udzial naglowkow odstajacych: {odstajace * 100:.1f}%")

    return {
        "corpus": data,
        "hard": hard_error_stats(data, topic_model, min_articles=min_articles, save=save),
        "soft": soft_error_stats(data, save=save),
        "topic_model": topic_model,
        "model_path": sciezka,
        "outlier_ratio": odstajace,
    }


def probability_profile_over_time(data, freq="MS"):
    """Miesieczny profil predykcji: srednie prawdopodobienstwo kazdej klasy i entropia.

    Entropia rozkladu (w bitach) mierzy niepewnosc klasyfikatora wobec
    pojedynczego naglowka: zero oznacza przypisanie calej masy jednemu
    wydawcy, log2(3) = 1,585 rozklad rownomierny. Srednia miesieczna pokazuje,
    czy model myli sie pewnie, czy z wahaniem.

    Zwraca DataFrame z kolumna date, kolumnami prawdopodobienstw i entropy.
    """
    kolumny_p = [f"probability_topic_{i}" for i in sorted(config.GAZETY)]
    d = data.dropna(subset=kolumny_p + ["date_parsed"]).copy()

    probs = d[kolumny_p].to_numpy(dtype=float)
    d["entropy"] = -np.sum(probs * np.log2(probs + 1e-9), axis=1)
    d["date"] = d["date_parsed"].dt.to_period("M")

    profil = d.groupby("date")[kolumny_p + ["entropy"]].mean().reset_index()
    profil["n"] = d.groupby("date").size().to_numpy()
    profil["date"] = profil["date"].dt.to_timestamp()
    return profil


def confidence_over_time(data, freq="MS"):
    """Srednie prawdopodobienstwo klasy PRAWDZIWEJ w miesiacu, osobno dla wydawcy.

    Miara czyta sie jako pewnosc modelu wobec tytulu: spadek oznacza, ze
    naglowki tego wydawcy staly sie dla klasyfikatora mniej rozpoznawalne.

    Zwraca DataFrame w postaci dlugiej: date, label, confidence, n.
    """
    d = data.dropna(subset=["date_parsed"]).copy()
    d["date"] = d["date_parsed"].dt.to_period("M")

    czesci = []
    for label in sorted(config.GAZETY):
        podzbior = d[d["label"] == label]
        kolumna = f"probability_topic_{label}"
        grupa = podzbior.groupby("date")[kolumna].agg(["mean", "size"]).reset_index()
        grupa.columns = ["date", "confidence", "n"]
        grupa["label"] = label
        czesci.append(grupa)

    conf = pd.concat(czesci, ignore_index=True)
    conf["date"] = conf["date"].dt.to_timestamp()
    return conf[["date", "label", "confidence", "n"]]


def keywords_by_period(data, months=6, top=8, min_docs=15, column="text"):
    """Czolowe terminy TF-IDF w kolejnych oknach czasowych o dlugosci months.

    Sluzy do zwiazania bledow klasyfikatora z wydarzeniami okresu: termin
    wchodzacy do czolowki w jednym oknie i znikajacy w nastepnym wskazuje
    temat, ktory przez chwile mylil model. Okna ponizej min_docs naglowkow sa
    pomijane, bo ranking TF-IDF liczony na kilkunastu dokumentach jest
    przypadkowy.

    Okna liczone sa z numeru miesiaca, a nie przez to_period("6M"): mnoznik
    przy czestosci miesiecznej nie skleja miesiecy w jeden okres, wiec
    grupowanie wychodzilo miesieczne mimo zadanej dlugosci okna.

    Zwraca DataFrame: wiersz to okno, kolumny to kolejne terminy.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from stop_words import get_stop_words

    stop = list(set(get_stop_words("pl")) | set(config.EXTRA_STOPWORDS))
    d = data.dropna(subset=["date_parsed", column]).copy()

    miesiac = d["date_parsed"].dt.year * 12 + (d["date_parsed"].dt.month - 1)
    poczatek = int(miesiac.min())
    d["okno"] = (miesiac - poczatek) // months

    wiersze = {}
    for okno in sorted(d["okno"].unique()):
        podzbior = d[d["okno"] == okno]
        if len(podzbior) < min_docs:
            continue
        od = pd.Period(year=(poczatek + int(okno) * months) // 12,
                       month=(poczatek + int(okno) * months) % 12 + 1, freq="M")
        opis = f"{od} - {od + months - 1}"

        wektoryzator = TfidfVectorizer(stop_words=stop, max_df=0.9, min_df=2, max_features=100)
        X = wektoryzator.fit_transform(podzbior[column].astype(str))
        wagi = X.sum(axis=0).A1
        terminy = wektoryzator.get_feature_names_out()
        czolowe = [terminy[i] for i in wagi.argsort()[::-1][:top]]
        wiersze[opis] = czolowe + [""] * (top - len(czolowe))

    return pd.DataFrame.from_dict(wiersze, orient="index",
                                  columns=[f"term {i + 1}" for i in range(top)])
