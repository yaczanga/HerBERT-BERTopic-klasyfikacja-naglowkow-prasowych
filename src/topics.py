# -*- coding: utf-8 -*-
"""
Modelowanie tematyczne BERTopic: budowa komponentow, funkcja celu Optuny
(single-objective NPMI), baseline'y LDA/NMF oraz redukcja outlierow.
Logika odpowiada notebookowi 01b. Ciezkie biblioteki importowane leniwie.
"""
import numpy as np

import config
from src.utils import enable_legacy_torch_load, topic_coherences, coherence_from_topics, rel

NGRAM_MAX = 3
MIN_TOPICS = 10

# Przestrzenie przeszukiwania obu strojen BERTopica, obok siebie. Korpus liczy
# okolo 56 tysiecy naglowkow, podzbior pomylek okolo 3 tysiecy, wiec rozne sa
# zakresy min_cluster_size i min_df oraz to, czy max_df jest strojone; reszta
# stosu jest wspolna. Trzymanie obu zakresow w jednym miejscu jest celowe:
# wczesniej byly wpisane osobno w dwie funkcje celu, przez co do opisu trafil
# zakres jednego strojenia obok konfiguracji z drugiego.
#
# Klucz to nazwa parametru w bazie prob Optuny. Zmiana nazwy zrywa wznawianie
# zapisanych badan, dlatego "csm" w przestrzeni pomylek zostaje jak bylo, mimo
# ze korpus uzywa pelnej nazwy. Kolejnosc wpisow tez ma znaczenie: wyznacza
# kolejnosc wywolan trial.suggest_*, a od niej zalezy ciag losowy samplera.
PRZESTRZEN_KORPUS = {
    "n_neighbors": ("int", 10, 50),
    "n_components": ("int", 5, 15),
    "min_dist": ("float", 0.0, 0.1),
    "min_cluster_size": ("int", 60, 120),
    "cluster_selection_method": ("cat", ["eom", "leaf"]),
    "min_df": ("int", 2, 10),
    "max_df": ("float", 0.7, 0.95),
}
PRZESTRZEN_POMYLKI = {
    "n_neighbors": ("int", 10, 40),
    "n_components": ("int", 5, 15),
    "min_dist": ("float", 0.0, 0.1),
    "min_cluster_size": ("int", 10, 40),
    "csm": ("cat", ["eom", "leaf"]),
    "min_df": ("int", 2, 5),
    "max_df": ("stala", 0.9),
}


def _sugeruj(trial, przestrzen):
    """Zamienia opis przestrzeni na slownik parametrow jednej proby Optuny."""
    params = {}
    for nazwa, opis in przestrzen.items():
        rodzaj = opis[0]
        if rodzaj == "int":
            params[nazwa] = trial.suggest_int(nazwa, opis[1], opis[2])
        elif rodzaj == "float":
            params[nazwa] = trial.suggest_float(nazwa, opis[1], opis[2])
        elif rodzaj == "cat":
            params[nazwa] = trial.suggest_categorical(nazwa, opis[1])
        else:
            params[nazwa] = opis[1]
    return params


def _zbuduj_model(params, tokenizer, embedding_model, representation_model,
                  ctfidf_model=None, calculate_probabilities=False):
    """Stos UMAP + HDBSCAN + BERTopic na gotowych parametrach.

    Jedyne miejsce, w ktorym ten stos powstaje. Obie funkcje celu i oba modele
    finalne roznia sie wylacznie wartosciami w params oraz dwiema flagami.
    Nazwe metody wyboru skupisk czyta spod "cluster_selection_method" albo
    "csm", bo tak nazywaja ja dwie bazy prob Optuny.

    UMAP nie dostaje low_memory - True jest jego wartoscia domyslna.
    """
    import hdbscan
    import umap
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import TfidfVectorizer

    umap_model = umap.UMAP(
        n_neighbors=int(params["n_neighbors"]),
        n_components=int(params["n_components"]),
        min_dist=float(params["min_dist"]),
        metric=params.get("metric_umap", "cosine"),
        random_state=config.RANDOM_SEED, n_jobs=-1)
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=int(params["min_cluster_size"]),
        cluster_selection_method=params.get("cluster_selection_method", params.get("csm")),
        metric=params.get("metric_hdbscan", "euclidean"),
        prediction_data=True, core_dist_n_jobs=-1)
    ngram = (1, int(params.get("ngram_range", NGRAM_MAX)))
    opcjonalne = {} if ctfidf_model is None else {"ctfidf_model": ctfidf_model}
    return BERTopic(
        embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model,
        vectorizer_model=TfidfVectorizer(ngram_range=ngram, tokenizer=tokenizer.tokenize),
        representation_model=representation_model, top_n_words=10,
        calculate_probabilities=calculate_probabilities, nr_topics=None, **opcjonalne)


def _dopasuj(model, texts, embeddings):
    """Dopasowuje model i zwraca (przypisania, liczba tematow, udzial odstajacych).

    Liczba tematow liczona jest z samych przypisan, a nie jako get_topic_info()
    minus jeden: ten drugi sposob zaniza wynik o jeden, gdy model nie zwrocil
    zadnej obserwacji odstajacej i w tabeli nie ma wiersza -1.
    """
    topics, _ = model.fit_transform(texts, embeddings=embeddings)
    topics = np.asarray(topics)
    n_topics = len(set(topics.tolist())) - (1 if -1 in topics else 0)
    return topics, n_topics, float((topics == -1).mean())


def _naloz_df(model, texts, tokenizer, min_df, max_df, ngram_max=NGRAM_MAX):
    """Naklada filtry min_df i max_df dopiero po wyznaczeniu tematow.

    Nalozone od razu odrzucilyby czesc slownika jeszcze przed klastrowaniem,
    a przy ostrym progu wektoryzator konczy sie bledem na pustym slowniku.
    Kolejnosc jest ta sama w funkcji celu i przy modelu finalnym, wiec model
    powstaje w warunkach, w ktorych byl oceniany.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    model.update_topics(texts, vectorizer_model=TfidfVectorizer(
        ngram_range=(1, int(ngram_max)), min_df=int(min_df), max_df=float(max_df),
        tokenizer=tokenizer.tokenize))


def build_components(device="cuda"):
    """Zwraca (tokenizer, embedding_model, ctfidf_model, representation_model)."""
    enable_legacy_torch_load()
    from nltk.tokenize import RegexpTokenizer
    from sentence_transformers import SentenceTransformer
    from bertopic.vectorizers import ClassTfidfTransformer
    from bertopic.representation import KeyBERTInspired

    tokenizer = RegexpTokenizer(r"\w+")
    embedding_model = SentenceTransformer(config.HERBERT_MODEL, device=device)
    return tokenizer, embedding_model, ClassTfidfTransformer(), KeyBERTInspired()


def make_objective(texts, embeddings, tokenizer, embedding_model, ctfidf_model, representation_model):
    """Tworzy funkcje celu Optuny (maksymalizacja NPMI) dla calego korpusu.

    Przestrzen przeszukiwania: PRZESTRZEN_KORPUS. df nakladane na koncu, spojnie
    z modelem finalnym; outliery logowane, nie karane.
    """
    def objective(trial):
        params = _sugeruj(trial, PRZESTRZEN_KORPUS)
        model = _zbuduj_model(params, tokenizer, embedding_model, representation_model,
                              ctfidf_model=ctfidf_model)
        _, n_topics, outlier_ratio = _dopasuj(model, texts, embeddings)
        trial.set_user_attr("n_topics", int(n_topics))
        trial.set_user_attr("outlier_ratio", outlier_ratio)
        if n_topics < MIN_TOPICS:
            print(f"  trial {trial.number}: n_topics={n_topics} (<{MIN_TOPICS}) -> odrzucone")
            return -1.0
        _naloz_df(model, texts, tokenizer, params["min_df"], params["max_df"])
        umass, uci, npmi = topic_coherences(model, texts, tokenizer)
        trial.set_user_attr("umass", umass)
        trial.set_user_attr("uci", uci)
        print(f"  trial {trial.number}: n_topics={n_topics}, outliers={outlier_ratio:.3f}, "
              f"NPMI={npmi:.4f}, UMass={umass:.3f}, UCI={uci:.3f}")
        return npmi

    return objective


def lda_nmf_baselines(texts, tokenizer, n_topics, top_n=10):
    """LDA + NMF na tym samym korpusie; ta sama metryka coherence co BERTopic.
    Zwraca DataFrame [model, UMass, UCI, NPMI]."""
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import NMF, LatentDirichletAllocation

    tok = tokenizer.tokenize
    cleaned = [tok(d.lower()) for d in texts]

    def topics_from(components, names):
        return [[names[i] for i in comp.argsort()[::-1][:top_n]] for comp in components]

    cv = CountVectorizer(tokenizer=tok, min_df=5, max_df=0.9)
    X_cv = cv.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=config.RANDOM_SEED,
                                    learning_method="online").fit(X_cv)
    lda_coh = coherence_from_topics(topics_from(lda.components_, cv.get_feature_names_out()), cleaned)

    tf = TfidfVectorizer(tokenizer=tok, min_df=5, max_df=0.9)
    X_tf = tf.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=config.RANDOM_SEED, init="nndsvd", max_iter=400).fit(X_tf)
    nmf_coh = coherence_from_topics(topics_from(nmf.components_, tf.get_feature_names_out()), cleaned)

    return pd.DataFrame([("LDA", *lda_coh), ("NMF", *nmf_coh)],
                        columns=["model", "UMass", "UCI", "NPMI"])


def finetuned_embeddings(texts, model_dir, batch_size=64, max_length=None, device="cuda"):
    """Mean-pooled embeddingi dokumentow z DOSTROJONEGO enkodera HerBERT.
    Uzywane, gdy chcemy modelowac tematy na reprezentacjach specyficznych dla zadania
    (np. analiza zle sklasyfikowanych naglowkow)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    max_length = max_length or config.MAX_LENGTH
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state              # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()   # (B, T, 1)
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(emb.cpu().numpy())
            print(f"  embeddingi: {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.vstack(out)


def misclassified_topic_model(texts, embeddings, n_trials=25, min_topics=5, device="cuda",
                              study_name="bertopic_misclassified", storage=None, resume=True):
    """Buduje model tematyczny BERTopic na zadanych tekstach + wstepnie policzonych
    embeddingach (scenariusz: zle sklasyfikowane naglowki, embeddingi fine-tuned).

    Optuna maksymalizuje NPMI, potem trenowany jest finalny model na najlepszych
    parametrach. Zwraca (model, results), gdzie results zawiera metryki i best_params.
    Logika odpowiada notebookowi 03.

    Proby trafiaja do bazy SQLite (config.MISCLASSIFIED_OPTUNA_DB), wiec przerwane
    strojenie da sie podjac. Przy resume=True kolejne wywolanie wczytuje istniejaca
    baze i DOKLADA n_trials prob do juz zapisanych, zamiast zaczynac od zera.
    Baza wiaze sie z konkretnym zbiorem tekstow: po przeliczeniu predykcji liczba
    pomylek sie zmienia i proby z dwoch przebiegow nie sa porownywalne, dlatego
    funkcja zapisuje n_docs w atrybutach study i ostrzega przy niezgodnosci.
    """
    enable_legacy_torch_load()
    import optuna
    from bertopic.vectorizers import ClassTfidfTransformer
    from bertopic.representation import KeyBERTInspired
    from nltk.tokenize import RegexpTokenizer
    from sentence_transformers import SentenceTransformer

    tokenizer = RegexpTokenizer(r"\w+")
    ctfidf = ClassTfidfTransformer()
    repr_model = KeyBERTInspired()
    # embedding_model tylko do osadzania slow przez KeyBERTInspired; dokumenty uzywaja
    # wstepnie policzonych embeddingow (arg embeddings=)
    word_embed = SentenceTransformer(config.HERBERT_MODEL, device=device)

    def objective(trial):
        params = _sugeruj(trial, PRZESTRZEN_POMYLKI)
        m = _zbuduj_model(params, tokenizer, word_embed, repr_model, ctfidf_model=ctfidf)
        _, n_topics, outlier = _dopasuj(m, texts, embeddings)
        trial.set_user_attr("n_topics", int(n_topics))
        trial.set_user_attr("outlier_ratio", outlier)
        if n_topics < min_topics:
            print(f"  trial {trial.number}: n_topics={n_topics} (<{min_topics}) -> odrzucone")
            return -1.0
        _naloz_df(m, texts, tokenizer, params["min_df"], params["max_df"])
        umass, uci, npmi = topic_coherences(m, texts, tokenizer)
        trial.set_user_attr("umass", umass)
        trial.set_user_attr("uci", uci)
        print(f"  trial {trial.number}: n_topics={n_topics}, outliers={outlier:.3f}, NPMI={npmi:.4f}")
        return npmi

    print("\nOptuna (BERTopic na bledach)...")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage or config.MISCLASSIFIED_OPTUNA_DB,
        load_if_exists=resume,
        direction="maximize",
    )
    zapisane = study.user_attrs.get("n_docs")
    if zapisane is None:
        study.set_user_attr("n_docs", len(texts))
    elif zapisane != len(texts):
        print(f"  UWAGA: baza prob powstala na {zapisane} naglowkach, teraz jest ich "
              f"{len(texts)}. Proby z obu przebiegow nie sa porownywalne - podaj inne "
              f"study_name albo skasuj baze.")
    print(f"  prob juz w bazie: {len(study.trials)}, dokladanych teraz: {n_trials}")
    study.optimize(objective, n_trials=n_trials)
    bp = study.best_trial
    print(f"\nNajlepsza proba #{bp.number}: NPMI={bp.value:.4f}, "
          f"tematy={bp.user_attrs['n_topics']}, outliery={bp.user_attrs['outlier_ratio']:.3f}")

    # finalny model na najlepszych parametrach
    model = _zbuduj_model(bp.params, tokenizer, word_embed, repr_model, ctfidf_model=ctfidf)
    _, n_topics, outlier = _dopasuj(model, texts, embeddings)
    _naloz_df(model, texts, tokenizer, bp.params["min_df"], PRZESTRZEN_POMYLKI["max_df"][1])
    umass, uci, npmi = topic_coherences(model, texts, tokenizer)

    results = {"n_docs": len(texts), "n_topics": int(n_topics), "outlier_ratio": outlier,
               "NPMI": npmi, "UMass": umass, "UCI": uci, "best_params": bp.params}
    return model, results


def load_topic_model(path, embedding_model=None):
    """Wczytuje zapisany model BERTopic, obchodzac niezgodnosc wektoryzatora.

    BERTopic zapisuje parametry wektoryzatora takie, jakie dostal, ale przy
    wczytywaniu odtwarza z nich zawsze CountVectorizer. Model zapisany
    z TfidfVectorizer (tak robi misclassified_topic_model) niesie wiec klucze
    norm, use_idf, smooth_idf i sublinear_tf, ktorych CountVectorizer nie zna,
    i wczytanie konczy sie TypeError. Tutaj te klucze sa tymczasowo odkladane
    na czas wczytania; nie wplywaja na tematy ani na ich reprezentacje, bo te
    sa juz policzone i zapisane w ctfidf.safetensors.
    """
    import json
    import shutil
    import tempfile
    from pathlib import Path

    from bertopic import BERTopic

    enable_legacy_torch_load()
    path = Path(path)
    cfg_path = path / "ctfidf_config.json"
    TYLKO_TFIDF = ("norm", "use_idf", "smooth_idf", "sublinear_tf")

    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        params = cfg.get("vectorizer_model", {}).get("params", {})
        kolidujace = [k for k in TYLKO_TFIDF if k in params]
        if kolidujace:
            # pracujemy na kopii, zeby nie zmieniac zapisanego modelu
            with tempfile.TemporaryDirectory() as tmp:
                kopia = Path(tmp) / path.name
                shutil.copytree(path, kopia)
                cfg2 = json.loads((kopia / "ctfidf_config.json").read_text(encoding="utf-8"))
                for k in kolidujace:
                    cfg2["vectorizer_model"]["params"].pop(k, None)
                (kopia / "ctfidf_config.json").write_text(
                    json.dumps(cfg2, ensure_ascii=False), encoding="utf-8")
                return BERTopic.load(str(kopia), embedding_model=embedding_model)

    return BERTopic.load(str(path), embedding_model=embedding_model)


def reduce_outliers_compare(topic_model, texts, tokenizer):
    """Stosuje reduce_outliers (c-TF-IDF) i zwraca DataFrame porownania przed/po
    (outlier_ratio + UMass/UCI/NPMI). Modyfikuje topic_model w miejscu."""
    import pandas as pd
    before = topic_coherences(topic_model, texts, tokenizer)
    before_out = float((np.array(topic_model.topics_) == -1).mean())
    new_topics = topic_model.reduce_outliers(texts, topic_model.topics_, strategy="c-tf-idf")
    topic_model.update_topics(texts, topics=new_topics)
    after = topic_coherences(topic_model, texts, tokenizer)
    after_out = float((np.array(topic_model.topics_) == -1).mean())
    return pd.DataFrame([("przed", before_out, *before), ("po", after_out, *after)],
                        columns=["etap", "outlier_ratio", "UMass", "UCI", "NPMI"])


def build_misclassified_topics(n_trials=25, probs_csv=None, model_dir=None, save=True,
                               study_name="bertopic_misclassified", storage=None, resume=True):
    """Etap 3 w calosci: model tematyczny na ZLE SKLASYFIKOWANYCH naglowkach testowych.

    Wybiera pomylki klasyfikatora, liczy dla nich embeddingi dostrojonym
    HerBERTem, stroi BERTopic Optuna i zapisuje wyniki. Wymaga GPU i trwa
    kilkadziesiat minut.

    save=False pomija zapis na dysk (przydatne przy probach).
    Zwraca (model, results, mis) - model BERTopic, slownik metryk, ramke
    naglowkow z przypisanym tematem w kolumnie mis_topic.
    """
    import json

    import pandas as pd

    probs_csv = probs_csv or config.PROBS_CSV
    model_dir = model_dir or config.HERBERT_DIR

    df = pd.read_csv(probs_csv)
    mis = df[(df["split"] == "test") & (df["label"] != df["predicted_label"])].reset_index(drop=True)
    texts = mis["text"].astype(str).tolist()

    embeddings = finetuned_embeddings(texts, str(model_dir))
    model, results = misclassified_topic_model(texts, embeddings, n_trials=n_trials,
                                               study_name=study_name, storage=storage,
                                               resume=resume)
    mis = mis.copy()
    mis["mis_topic"] = model.topics_

    if save:
        with open(config.DATA / "misclassified_topics_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=float)
        model.save(str(config.MISCLASSIFIED_DIR), serialization="safetensors", save_ctfidf=True)
        mis.to_csv(config.DATA / "misclassified_with_topics.csv", index=False)

    return model, results, mis


def load_saved_model(embedding_model=None, device="cuda", prefer_merged=True):
    """Wczytuje zapisany model BERTopic i zwraca (model, sciezka).

    Kolejnosc: model scalony (config.BERTOPIC_MERGED), a gdy go nie ma - bazowy
    (config.BERTOPIC_DIR). Gdy nie ma zadnego, zglasza blad: tematy musi zapisac
    notebook 01b, bo scalanie zalezy od konkretnego przebiegu strojenia
    i odtworzenie go z numerow tematow dawaloby inny podzial.
    """
    kandydaci = [config.BERTOPIC_MERGED, config.BERTOPIC_DIR]
    if not prefer_merged:
        kandydaci.reverse()

    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        enable_legacy_torch_load()
        embedding_model = SentenceTransformer(config.HERBERT_MODEL, device=device)

    for sciezka in kandydaci:
        if sciezka.exists():
            return load_topic_model(str(sciezka), embedding_model=embedding_model), sciezka

    raise FileNotFoundError(
        f"Nie znaleziono zapisanego modelu BERTopic w {config.BERTOPIC_MERGED} "
        f"ani w {config.BERTOPIC_DIR}. Uruchom najpierw notebook 01b.")


def assign_topics(texts, topic_model, cache_csv=None, batch_size=500, force=False):
    """Przypisuje tematy do tekstow, z pamiecia wyniku na dysku.

    Transform na kilkudziesieciu tysiacach naglowkow trwa kilka minut i wymaga
    GPU, wiec wynik laduje do cache_csv i jest wczytywany przy kolejnych
    przebiegach. Cache uznajemy za wazny tylko wtedy, gdy liczba wierszy zgadza
    sie z liczba tekstow - inaczej liczymy od nowa.

    Zwraca (topics, probs); probs jest None, gdy wynik przyszedl z cache.
    """
    import pandas as pd

    from src.utils import assign_topics_batched

    cache_csv = cache_csv or (config.DATA / "valid_data_with_topics.csv")
    if not force and cache_csv.exists():
        zapisane = pd.read_csv(cache_csv)
        if len(zapisane) == len(texts):
            print(f"Tematy z cache: {cache_csv.name} ({len(zapisane)} wierszy)")
            return zapisane["topic"].tolist(), None
        print(f"Cache {cache_csv.name} ma {len(zapisane)} wierszy wobec "
              f"{len(texts)} tekstow - przeliczam tematy od nowa.")

    topics, probs = assign_topics_batched(topic_model, texts, batch_size=batch_size)
    pd.DataFrame({"topic": topics}).to_csv(cache_csv, index=False)
    print(f"Zapisano przypisania tematow -> {cache_csv.name}")
    return topics, probs


def outlier_share_by_cluster_size(study, buckets=((60, 80), (80, 100), (100, 120))):
    """Udzial obserwacji odstajacych w probach strojenia wzgledem min_cluster_size.

    Odpowiada na pytanie, czy wysokie pokrycie odstajacymi zbija sam prog
    wielkosci skupiska. Do zestawienia wchodza proby uzyteczne, czyli te
    z n_topics >= MIN_TOPICS; pozostale funkcja celu odrzuca wartoscia -1.
    Zwraca (DataFrame po kubelkach, slownik z liczba prob, zakresem i korelacja).
    """
    import pandas as pd

    uzyteczne = [
        (t.user_attrs["outlier_ratio"], float(t.params["min_cluster_size"]))
        for t in study.trials
        if t.state.name == "COMPLETE"
        and t.user_attrs.get("n_topics", 0) >= MIN_TOPICS
        and "outlier_ratio" in t.user_attrs
        and "min_cluster_size" in t.params
    ]
    if not uzyteczne:
        raise ValueError("Baza prob nie zawiera prob uzytecznych z zapisanym udzialem odstajacych.")

    udzialy = np.array([u for u, _ in uzyteczne])
    progi = np.array([p for _, p in uzyteczne])

    wiersze = []
    for lo, hi in buckets:
        maska = (progi >= lo) & (progi < hi)
        if maska.any():
            wiersze.append({"min_cluster_size": f"{lo}-{hi}",
                            "proby": int(maska.sum()),
                            "mediana_odstajacych": float(np.median(udzialy[maska]))})

    podsumowanie = {
        "proby_uzyteczne": len(uzyteczne),
        "proby_odrzucone": sum(1 for t in study.trials
                               if t.state.name == "COMPLETE"
                               and t.user_attrs.get("n_topics", 0) < MIN_TOPICS),
        "udzial_min": float(udzialy.min()),
        "udzial_max": float(udzialy.max()),
        "korelacja_z_progiem": float(np.corrcoef(udzialy, progi)[0, 1]),
    }
    return pd.DataFrame(wiersze), podsumowanie


def best_params_from_study(study, save=True, out_json=None):
    """Parametry najlepszej proby Optuny w formacie przyjmowanym przez build_final_model.

    Uzupelnia o stale, ktorych funkcja celu nie przeszukuje (metryki UMAP
    i HDBSCAN, gorny rzad n-gramu), zeby budowa modelu finalnego nie powtarzala
    ich z pamieci. Zapisuje komplet do JSON razem z wynikiem proby.
    """
    import json

    proba = study.best_trial
    bp = proba.params
    params = {
        "n_neighbors": bp["n_neighbors"],
        "n_components": bp["n_components"],
        "metric_umap": "cosine",
        "min_dist": bp["min_dist"],
        "min_cluster_size": bp["min_cluster_size"],
        "metric_hdbscan": "euclidean",
        "cluster_selection_method": bp["cluster_selection_method"],
        "ngram_range": NGRAM_MAX,
        "min_df": bp["min_df"],
        "max_df": bp["max_df"],
    }
    if save:
        out_json = out_json or (config.DATA / "bertopic_optuna_best_params.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"best_trial": proba.number, "npmi": proba.value,
                       "outlier_ratio": proba.user_attrs.get("outlier_ratio"),
                       "n_topics": proba.user_attrs.get("n_topics"),
                       "params": params}, f, indent=2, ensure_ascii=False)
    return params


def load_best_params(path=None):
    """Parametry BERTopic zapisane przez best_params_from_study."""
    import json
    path = path or (config.DATA / "bertopic_optuna_best_params.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("params", {})


def build_final_model(params, texts, embeddings, tokenizer, embedding_model,
                      representation_model, save=True, out_dir=None):
    """Buduje model BERTopic na najlepszych parametrach, zapisuje go i zwraca.

    O kolejnosci filtrow df wzgledem klastrowania - patrz _naloz_df.
    """
    import pickle

    model = _zbuduj_model(params, tokenizer, embedding_model, representation_model,
                          calculate_probabilities=True)
    model.fit_transform(texts, embeddings=embeddings)
    _naloz_df(model, texts, tokenizer, params["min_df"], params["max_df"],
              ngram_max=params["ngram_range"])

    if save:
        out_dir = out_dir or config.BERTOPIC_DIR
        model.save(str(out_dir), serialization="safetensors", save_ctfidf=True)
        with open(config.DATA / "representative_docs.pkl", "wb") as f:
            pickle.dump(model.representative_docs_, f)
        print(f"Zapisano model -> {rel(out_dir)}")
    return model


def merge_level_search(topic_model, texts, tokenizer, save=True, out_pkl=None):
    """Przebiega kolejne poziomy scalania hierarchicznego i mierzy spojnosc.

    Dla kazdego dystansu w hierarchii skleja rozlaczne grupy tematow i liczy
    UMass, UCI i NPMI tym samym gensimowym kodem co reszta projektu
    (src.utils.coherence_from_topics). Kolumna sum_normalised to suma trzech
    miar przeskalowanych min-max; sluzy do wyboru poziomu scalania, bo same
    miary maja rozne zakresy i nie daja sie dodac wprost.

    Kazdy poziom wymaga glebokiej kopii modelu, wiec przebieg jest kosztowny
    pamieciowo; po kazdym kroku kopia jest usuwana, a cache GPU czyszczony.

    Zwraca (DataFrame wynikow, ramke hierarchii).
    """
    import copy
    import gc
    import pickle

    import pandas as pd

    from src.utils import coherence_from_topics

    analyzer = tokenizer.tokenize if hasattr(tokenizer, "tokenize") else tokenizer
    cleaned = [analyzer(doc) for doc in texts]

    try:
        import torch
        pusty_cache = torch.cuda.empty_cache
    except Exception:
        def pusty_cache():
            return None

    hierarchia = topic_model.hierarchical_topics(texts)
    wyniki = []
    out_pkl = out_pkl or (config.OUTPUTS_DIR / "merger_optimisation_results.pkl")

    for dystans in sorted(hierarchia["Distance"].unique()):
        kopia = None
        try:
            kandydaci, uzyte = [], set()
            for grupa in sorted(hierarchia.loc[hierarchia["Distance"] <= dystans, "Topics"].tolist(),
                                key=len, reverse=True):
                if not (set(grupa) & uzyte):
                    kandydaci.append(grupa)
                    uzyte |= set(grupa)
            if len(kandydaci) < 2:
                kandydaci = [t for grupa in kandydaci for t in grupa]

            kopia = copy.deepcopy(topic_model)
            kopia.merge_topics(docs=texts, topics_to_merge=kandydaci)

            info = kopia.get_topic_info()
            slowa = []
            for temat in info.Topic:
                if temat == -1:
                    continue
                w = [s for s, _ in kopia.get_topic(temat)]
                if len(w) >= 2:
                    slowa.append(w)
            umass, uci, npmi = coherence_from_topics(slowa, cleaned)

            wyniki.append({"distance": dystans,
                           "n_topics": int((info.Topic != -1).sum()),
                           "merge_groups": kandydaci,
                           "umass": umass, "uci": uci, "npmi": npmi})
            print(f"  dystans {dystans:.4f}: {wyniki[-1]['n_topics']} tematow, "
                  f"NPMI={npmi:.4f}, UMass={umass:.3f}, UCI={uci:.3f}")

            if save:
                with open(out_pkl, "wb") as f:
                    pickle.dump(wyniki, f)
        except Exception as blad:
            print(f"Dalsze scalanie niemozliwe (dystans={dystans:.4f}): {blad}")
            break
        finally:
            del kopia
            gc.collect()
            pusty_cache()

    tab = pd.DataFrame(wyniki)
    if len(tab):
        miary = ["umass", "uci", "npmi"]
        znorm = tab[miary].apply(lambda k: (k - k.min()) / (k.max() - k.min()))
        tab["sum_normalised"] = znorm.sum(axis=1)
    print(f"Przebadano {len(tab)} poziomow scalania.")
    return tab, hierarchia


def apply_merge_level(topic_model, texts, results, row=None, save=True, out_dir=None):
    """Scala tematy na wybranym poziomie i zapisuje model scalony.

    row wskazuje wiersz z merge_level_search; gdy jest None, wybierany jest
    poziom o najwyzszym sum_normalised. Wczesniej poziom byl wpisany w notebooku
    numerem wiersza, ktory zalezal od konkretnego przebiegu i nie dawal sie
    odtworzyc - teraz kryterium jest jawne, a wybrany poziom wypisywany.

    Dziala na kopii, wiec model wejsciowy zostaje nienaruszony. Zwraca model
    scalony.
    """
    import copy

    if row is None:
        row = int(results["sum_normalised"].idxmax())
    wiersz = results.loc[row]
    print(f"Poziom scalania: wiersz {row}, dystans {wiersz['distance']:.4f}, "
          f"{wiersz['n_topics']} tematow, sum_normalised {wiersz['sum_normalised']:.3f}")

    scalony = copy.deepcopy(topic_model)
    scalony.merge_topics(docs=texts, topics_to_merge=wiersz["merge_groups"])

    if save:
        out_dir = out_dir or config.BERTOPIC_MERGED
        scalony.save(str(out_dir), serialization="safetensors", save_ctfidf=True)
        print(f"Zapisano model scalony -> {rel(out_dir)}")
    return scalony


def compare_embedding_backbones(texts, params, tokenizer, representation_model,
                                candidates=None, device="cuda"):
    """Porownuje modele osadzen: liczba tematow, udzial outlierow i spojnosc.

    Surowy HerBERT uzywany jest tu jako enkoder zdaniowy, do czego nie byl
    uczony, co moze wspolodpowiadac za wysoki udzial obserwacji odstajacych.
    Funkcja sprawdza to, przeliczajac te sama konfiguracje na innych
    enkoderach. Kazdy kandydat wymaga policzenia embeddingow od nowa.

    Zwraca DataFrame [embedding_model, n_topics, outlier_ratio, UMass, UCI, NPMI].
    """
    import hdbscan
    import pandas as pd
    import umap
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    from src.utils import topic_coherences

    candidates = candidates or [
        config.HERBERT_MODEL,
        "sdadas/st-polish-paraphrase-from-mpnet-base-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ]

    wiersze = []
    for nazwa in candidates:
        enkoder = SentenceTransformer(nazwa, device=device)
        embeddings = enkoder.encode(texts, show_progress_bar=True)
        model = BERTopic(
            embedding_model=enkoder,
            umap_model=umap.UMAP(n_neighbors=int(params["n_neighbors"]),
                                 n_components=int(params["n_components"]),
                                 min_dist=float(params["min_dist"]), metric="cosine",
                                 random_state=config.RANDOM_SEED, n_jobs=-1),
            hdbscan_model=hdbscan.HDBSCAN(
                min_cluster_size=int(params["min_cluster_size"]),
                cluster_selection_method=params["cluster_selection_method"],
                metric="euclidean", prediction_data=True),
            vectorizer_model=TfidfVectorizer(ngram_range=(1, int(params["ngram_range"])),
                                             tokenizer=tokenizer.tokenize),
            representation_model=representation_model,
            calculate_probabilities=False, nr_topics=None)
        tematy, _ = model.fit_transform(texts, embeddings=embeddings)
        tematy = np.asarray(tematy)
        outliery = float((tematy == -1).mean())
        n_tematow = len(set(tematy.tolist())) - (1 if -1 in tematy else 0)
        spojnosc = topic_coherences(model, texts, tokenizer)
        wiersze.append((nazwa, n_tematow, outliery, *spojnosc))
        print(f"{nazwa}: n_topics={n_tematow}, outliers={outliery:.3f}, NPMI={spojnosc[2]:.4f}")

    return pd.DataFrame(wiersze, columns=["embedding_model", "n_topics", "outlier_ratio",
                                          "UMass", "UCI", "NPMI"])
