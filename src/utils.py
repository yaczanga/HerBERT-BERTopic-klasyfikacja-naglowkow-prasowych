# -*- coding: utf-8 -*-
"""
Wspolne funkcje pomocnicze dla wszystkich notebookow projektu.

Cel: jedno zrodlo prawdy dla rzeczy, ktore wczesniej byly kopiowane miedzy
notebookami (obejscie ladowania modeli, odczyt podzialu, batchowane
przypisywanie tematow, czyszczenie naglowkow, spojnosc NPMI).
"""

import os
import re
import gc
import json

import numpy as np

import config

# Alias mapowania etykiet na nazwy gazet. Definicja stoi w config.py; tutaj jest
# tylko nazwa, zeby notebooki importujace z src.utils nie trafialy na druga kopie.
GAZETY = config.GAZETY

SPLIT_INFO_PATH = str(config.SPLIT_INFO)


def enable_legacy_torch_load():
    """Pozwala wczytywac starsze checkpointy (HerBERT/BERTopic) mimo restrykcji
    nowszego PyTorcha/transformers. Ustawia flagi srodowiskowe i lagodzi kontrole
    bezpieczenstwa ladowania wag. Wywolac raz, na poczatku notebooka."""
    os.environ.setdefault("HF_ALLOW_INSECURE_TORCH_LOAD", "1")
    os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import transformers
        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass


def load_split_dates(path=SPLIT_INFO_PATH, fallback=("2024-02-04", "2025-02-04")):
    """Zwraca (split_train_date, split_val_date) jako pandas.Timestamp.
    split_train_date = granica TRAIN/VALID, split_val_date = granica VALID/TEST.
    Gdy plik nie istnieje lub jest niekompletny, uzywa wartosci zapasowych."""
    import pandas as pd
    train_s, val_s = fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
        train_s = info.get("split_train_date", info.get("split_date", train_s))
        val_s = info.get("split_val_date", info.get("split_date", val_s))
    except Exception:
        pass
    return pd.to_datetime(train_s), pd.to_datetime(val_s)


def clean_headline(s):
    """Usuwa redakcyjne dopiski [..] (np. [WIDEO], [ROZMOWA]), zwija biale znaki
    i porzadkuje brzegi. Zachowuje tresciowe prefiksy (USA:, MSZ:) i cytaty."""
    s = str(s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" :–-|•\t")
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    return s.strip()


def lemmatize_headline(text, morfeusz, stopwords=()):
    """Lematyzuje naglowek analizatorem Morfeusz 2 i odsiewa slowa funkcyjne.

    Dla kazdej pozycji w tekscie bierze PIERWSZA interpretacje morfologiczna
    (Morfeusz zwraca graf, wiec pozycji bywa kilka na slowo). Zostawia lematy
    dluzsze niz dwa znaki, zlozone z samych liter i spoza listy stopwords.

    morfeusz   - instancja morfeusz2.Morfeusz
    stopwords  - zbior slow do odsiania; domyslnie pusty
    Zwraca liste lematow; przy bledzie analizy zwraca pusta liste.
    """
    stop = set(stopwords)
    try:
        lemmas, next_node = [], 0
        for start, end, interp in morfeusz.analyse(str(text)):
            if start != next_node:          # pomijamy alternatywne sciezki w grafie
                continue
            lemma = interp[1].split(":")[0].lower()
            if lemma not in stop and len(lemma) > 2 and lemma.isalpha():
                lemmas.append(lemma)
            next_node = end
        return lemmas
    except Exception:
        return []


def topic_label(topic_model, topic_id, n_words=2, fallback="Temat {}"):
    """Krotka etykieta tematu z dwoch czolowych slow jego reprezentacji.

    Sluzy do podpisywania serii na wykresach, gdzie numer tematu nic nie mowi.
    Gdy temat nie istnieje albo nie ma reprezentacji, zwraca etykiete zastepcza
    z numerem liczonym od jedynki.
    """
    try:
        info = topic_model.get_topic_info()
        words = info.loc[info["Topic"] == topic_id, "Representation"].iloc[0]
        etykieta = ", ".join(words[:n_words])
        return etykieta or fallback.format(topic_id + 1)
    except Exception:
        return fallback.format(topic_id + 1)


def assign_topics_batched(topic_model, texts, batch_size=500, verbose=True):
    """Przypisuje tematy w paczkach (oszczedza pamiec GPU). Zwraca (topics, probs).
    Po kazdej paczce czysci cache CUDA/RAM."""
    try:
        import torch
        _empty = torch.cuda.empty_cache
    except Exception:
        def _empty():
            return None

    all_topics, all_probs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        b_topics, b_probs = topic_model.transform(batch)
        all_topics.extend(b_topics)
        if b_probs is not None:
            if isinstance(b_probs, list):
                all_probs.extend(b_probs)
            elif len(all_probs) == 0:
                all_probs = b_probs
            else:
                all_probs = np.vstack((all_probs, b_probs))
        _empty()
        gc.collect()
        if verbose:
            print(f"  Przetworzono {min(i + batch_size, len(texts))} z {len(texts)} dokumentow...")
    return all_topics, all_probs


def coherence_from_topics(topics_words, cleaned_docs):
    """Liczy (UMass, UCI, NPMI) dla DOWOLNEJ listy tematow (kazdy temat = lista slow),
    na danym, juz stokenizowanym korpusie. Pozwala porownywac BERTopic, LDA i NMF
    ta sama metryka i na tym samym korpusie referencyjnym.
    Zwraca (umass, uci, npmi); brakujace/bledne wartosci jako -1.0."""
    import gensim.corpora as corpora
    from gensim.models.coherencemodel import CoherenceModel

    id2word = corpora.Dictionary(cleaned_docs)
    corpus = [id2word.doc2bow(doc) for doc in cleaned_docs]
    safe = [[w for w in words if w in id2word.token2id] for words in topics_words]
    safe = [w for w in safe if len(w) >= 2]
    if not safe:
        return -1.0, -1.0, -1.0

    def _coh(metric, **kw):
        try:
            return float(CoherenceModel(topics=safe, dictionary=id2word,
                                        coherence=metric, **kw).get_coherence())
        except Exception:
            return -1.0

    return (_coh("u_mass", corpus=corpus),
            _coh("c_uci", texts=cleaned_docs),
            _coh("c_npmi", texts=cleaned_docs))


def topic_coherences(topic_model, texts, tokenizer):
    """UMass/UCI/NPMI dla dopasowanego modelu BERTopic (pomija temat -1)."""
    info = topic_model.get_topic_info()
    topics_words = []
    for t in [tt for tt in info.Topic if tt != -1]:
        words = [w for w, _ in topic_model.get_topic(t)]
        if words:
            topics_words.append(words)
    if not topics_words:
        return -1.0, -1.0, -1.0
    analyzer = tokenizer.tokenize if hasattr(tokenizer, "tokenize") else tokenizer
    cleaned = [analyzer(doc) for doc in texts]
    return coherence_from_topics(topics_words, cleaned)


def rel(path):
    """Sciezka wzgledem korzenia projektu, do wydrukow w notebookach.

    Wydruk sciezki bezwzglednej zapisuje uklad dysku autora w wyniku komorki,
    ktory trafia do repozytorium. Postac wzgledna niesie te sama informacje
    i czyta sie tak samo u kazdego.
    """
    from pathlib import Path
    try:
        import config as _cfg
        return Path(path).resolve().relative_to(_cfg.ROOT).as_posix()
    except (ValueError, ImportError):
        return str(path)

