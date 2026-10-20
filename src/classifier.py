# -*- coding: utf-8 -*-
"""
Klasyfikator HerBERT: budulec treningu (strata wazona + focal + warstwowe LR),
metryki, wagi klas, tokenizacja oraz proste baseline'y.
Logika odpowiada notebookowi 01a.

Uwaga: ciezkie zaleznosci (torch/transformers) importowane sa leniwie wewnatrz
funkcji/klas, zeby modul dalo sie zaimportowac take bez GPU (np. dla baseline'ow).
"""
from pathlib import Path

import numpy as np

import config


# ----------------------------------------------------------------------------
# Metryki i wagi klas
# ----------------------------------------------------------------------------
def compute_metrics(pred, gazety=None):
    """compute_metrics dla Trainer-a: accuracy, F1/precision/recall (macro + per klasa)."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    gazety = gazety or config.GAZETY
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    metrics = {"accuracy": acc, "f1_macro": f1_m, "precision_macro": p_m, "recall_macro": r_m}
    for i, name in gazety.items():
        metrics[f"f1_{name}"] = f1[i]
        metrics[f"precision_{name}"] = p[i]
        metrics[f"recall_{name}"] = r[i]
    return metrics


def compute_class_weights(train_labels):
    """Wagi klas metoda 'balanced' jako torch.Tensor."""
    import torch
    from sklearn.utils.class_weight import compute_class_weight
    w = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    return torch.tensor(w, dtype=torch.float)


# ----------------------------------------------------------------------------
# Tokenizacja
# ----------------------------------------------------------------------------
def make_tokenize_function(tokenizer, max_length=None):
    max_length = max_length or config.MAX_LENGTH

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True, max_length=max_length)
    return tokenize_function


# ----------------------------------------------------------------------------
# Warstwowe tempo uczenia + zamrazanie warstw
# ----------------------------------------------------------------------------
def get_optimizer_grouped_parameters(model, base_lr, decay_rate, weight_decay):
    """Grupy parametrow z malejacym w glab sieci tempem uczenia (layer-wise LR decay)."""
    no_decay = ["bias", "LayerNorm.weight"]
    num_layers = getattr(model.config, "num_hidden_layers", 12)
    groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        if "classifier" in name:
            mult = 1.0
        elif "pooler" in name:
            mult = decay_rate
        elif "encoder.layer." in name:
            try:
                parts = name.split(".")
                layer_idx = int(parts[parts.index("layer") + 1])
                mult = decay_rate ** (num_layers - layer_idx)
            except (ValueError, IndexError):
                mult = decay_rate
        elif "embeddings" in name:
            mult = decay_rate ** (num_layers + 1)
        else:
            mult = decay_rate
        groups.append({"params": [param], "weight_decay": wd, "lr": base_lr * mult})
    return groups


def freeze_model_layers(model, num_frozen_layers):
    """Zamraza embeddingi + num_frozen_layers najnizszych warstw enkodera."""
    if num_frozen_layers <= 0:
        return
    base = getattr(model, "bert", None) or getattr(model, "roberta", None)
    if base is None:
        return
    if hasattr(base, "embeddings"):
        for p in base.embeddings.parameters():
            p.requires_grad = False
    encoder = getattr(base, "encoder", None)
    if encoder is not None and hasattr(encoder, "layer"):
        for i in range(min(num_frozen_layers, len(encoder.layer))):
            for p in encoder.layer[i].parameters():
                p.requires_grad = False


def build_weighted_loss_trainer_cls():
    """Fabryka klasy WeightedLossTrainer (importuje transformers leniwie)."""
    import torch
    import torch.nn.functional as F
    from transformers import Trainer

    class WeightedLossTrainer(Trainer):
        def __init__(self, class_weights=None, focal_loss_gamma=None, decay_rate=None,
                     *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights
            self.focal_loss_gamma = focal_loss_gamma
            self.decay_rate = decay_rate

        def create_optimizer(self):
            if self.decay_rate is not None and self.decay_rate < 1.0:
                if self.optimizer is None:
                    groups = get_optimizer_grouped_parameters(
                        self.model, self.args.learning_rate, self.decay_rate,
                        self.args.weight_decay)
                    cls, kw = Trainer.get_optimizer_cls_and_kwargs(self.args)
                    self.optimizer = cls(groups, **kw)
                return self.optimizer
            return super().create_optimizer()

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            ls = getattr(self.args, "label_smoothing_factor", 0.0)
            if self.focal_loss_gamma is not None and self.focal_loss_gamma > 0:
                logits_r = logits.view(-1, self.model.config.num_labels)
                labels_r = labels.view(-1)
                ce = F.cross_entropy(logits_r, labels_r, reduction="none", label_smoothing=ls)
                p = torch.exp(-ce)
                focal = ((1 - p) ** self.focal_loss_gamma) * ce
                if self.class_weights is not None:
                    w = self.class_weights.to(model.device)
                    focal = focal * w[labels_r]
                loss = focal.mean()
            else:
                if self.class_weights is not None:
                    loss_fct = torch.nn.CrossEntropyLoss(
                        weight=self.class_weights.to(model.device), label_smoothing=ls)
                else:
                    loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=ls)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


# ----------------------------------------------------------------------------
# Strojenie hiperparametrow (Optuna)
# ----------------------------------------------------------------------------
def build_pruning_callback_cls():
    """Fabryka klasy OptunaPruningCallback (importuje transformers leniwie).

    Callback raportuje metryke po kazdej ewaluacji i przerywa probe, gdy pruner
    uzna ja za nierokujaca. Raportuje po `global_step`, bo ewaluacja idzie co
    N krokow, nie co epoke.
    """
    import transformers
    from optuna.exceptions import TrialPruned

    class OptunaPruningCallback(transformers.TrainerCallback):
        def __init__(self, trial, monitor="eval_f1_macro"):
            self.trial = trial
            self.monitor = monitor

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None:
                return
            value = metrics.get(self.monitor)
            if value is None:
                return
            self.trial.report(value, step=state.global_step)
            if self.trial.should_prune():
                raise TrialPruned()

    return OptunaPruningCallback


# Przestrzen przeszukiwania klasyfikatora - jedno zrodlo prawdy dla strojenia.
SEARCH_SPACE = {
    "learning_rate": ("float", 1e-5, 3e-5, {"log": True}),
    "num_train_epochs": ("int", 3, 5, {}),
    "weight_decay": ("float", 0.01, 0.4, {}),
    "warmup_ratio": ("float", 0.0, 0.1, {}),
    "hidden_dropout_prob": ("float", 0.1, 0.3, {}),
    "attention_probs_dropout_prob": ("float", 0.1, 0.3, {}),
    "label_smoothing_factor": ("float", 0.0, 0.15, {}),
    "per_device_train_batch_size": ("categorical", [8, 16, 32], None, {}),
    "focal_loss_gamma": ("float", 1.0, 3.0, {}),
    "decay_rate": ("float", 0.8, 0.95, {}),
    "lr_scheduler_type": ("categorical", ["linear", "cosine"], None, {}),
    "num_frozen_layers": ("int", 0, 8, {}),
    "adam_beta2": ("categorical", [0.999, 0.98, 0.99], None, {}),
}

TARGET_BATCH_SIZE = 32   # efektywna wielkosc wsadu, utrzymywana akumulacja gradientu


def suggest_params(trial):
    """Losuje komplet hiperparametrow z SEARCH_SPACE dla jednej proby Optuny."""
    out = {}
    for nazwa, (kind, a, b, kw) in SEARCH_SPACE.items():
        if kind == "float":
            out[nazwa] = trial.suggest_float(nazwa, a, b, **kw)
        elif kind == "int":
            out[nazwa] = trial.suggest_int(nazwa, a, b)
        else:
            out[nazwa] = trial.suggest_categorical(nazwa, a)
    return out


def make_optuna_objective(tokenized_train, tokenized_valid, class_weights,
                          model_name=None, output_dir=None, metric="eval_f1_macro",
                          eval_steps=150, verbose=True):
    """Buduje funkcje celu Optuny dla fine-tuningu HerBERTa.

    Zwracana funkcja przyjmuje `trial` i oddaje MAKSIMUM metryki po wszystkich
    ewaluacjach w obrebie proby. Jest to statystyka selekcyjna obciazona w gore:
    wartosc modelu finalnego bywa nizsza i tak nalezy ja raportowac.
    """
    import torch
    from optuna.exceptions import TrialPruned
    from transformers import AutoConfig, AutoModelForSequenceClassification, TrainingArguments

    model_name = model_name or config.HERBERT_MODEL
    output_dir = str(output_dir or (config.MODELS / "herbert_optuna"))
    WeightedLossTrainer = build_weighted_loss_trainer_cls()
    PruningCallback = build_pruning_callback_cls()

    def objective(trial):
        p = suggest_params(trial)
        grad_accum = max(1, TARGET_BATCH_SIZE // p["per_device_train_batch_size"])
        if verbose:
            print(f"\nProba {trial.number}: lr={p['learning_rate']:.2e}, "
                  f"epok={p['num_train_epochs']}, wsad={p['per_device_train_batch_size']}"
                  f" x{grad_accum}, gamma={p['focal_loss_gamma']:.2f}")

        cfg = AutoConfig.from_pretrained(model_name, num_labels=3)
        cfg.hidden_dropout_prob = p["hidden_dropout_prob"]
        cfg.attention_probs_dropout_prob = p["attention_probs_dropout_prob"]
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
        freeze_model_layers(model, p["num_frozen_layers"])

        args = TrainingArguments(
            output_dir=f"{output_dir}/trial_{trial.number}",
            fp16=torch.cuda.is_available(),
            num_train_epochs=p["num_train_epochs"],
            per_device_train_batch_size=p["per_device_train_batch_size"],
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=grad_accum,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="no",
            load_best_model_at_end=False,
            learning_rate=p["learning_rate"],
            weight_decay=p["weight_decay"],
            warmup_ratio=p["warmup_ratio"],
            label_smoothing_factor=p["label_smoothing_factor"],
            metric_for_best_model=metric.replace("eval_", ""),
            greater_is_better=True,
            logging_steps=eval_steps,
            disable_tqdm=True,
            report_to="none",
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            lr_scheduler_type=p["lr_scheduler_type"],
            adam_beta2=p["adam_beta2"],
        )

        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            focal_loss_gamma=p["focal_loss_gamma"],
            decay_rate=p["decay_rate"],
            model=model, args=args,
            train_dataset=tokenized_train, eval_dataset=tokenized_valid,
            compute_metrics=compute_metrics,
            callbacks=[PruningCallback(trial, monitor=metric)],
        )

        try:
            trainer.train()
        except TrialPruned:
            del model, trainer
            torch.cuda.empty_cache()
            raise

        wynik = trainer.evaluate()
        historia = [log[metric] for log in trainer.state.log_history if metric in log]
        wartosc = max(historia) if historia else wynik.get(metric, 0.0)
        if verbose:
            print(f"  proba {trial.number}: najlepsze {metric}={wartosc:.4f} "
                  f"(ostatni krok {wynik.get(metric, 0.0):.4f})")
        del model, trainer
        torch.cuda.empty_cache()
        return wartosc

    return objective


def tune_hyperparameters(tokenized_train, tokenized_valid, class_weights, n_trials=150,
                         study_name="herbert_finetune", storage=None, resume=True,
                         metric="eval_f1_macro", model_name=None, output_dir=None,
                         save=True):
    """Strojenie hiperparametrow Optuna z medianowym prunerem. Wymaga GPU.

    Zapisuje najlepsze parametry do data/herbert_optuna_best_params.json oraz
    tabele prob do herbert_optuna_all_trials.csv. Zwraca obiekt study.
    """
    import json

    import optuna

    output_dir = str(output_dir or (config.MODELS / "herbert_optuna"))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage or config.HERBERT_OPTUNA_DB,
        load_if_exists=resume,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1),
    )
    objective = make_optuna_objective(tokenized_train, tokenized_valid, class_weights,
                                      model_name=model_name, output_dir=output_dir,
                                      metric=metric)
    study.optimize(objective, n_trials=n_trials)

    if save and study.best_trial is not None:
        best = {"best_trial_number": study.best_trial.number,
                "best_value": study.best_value, "metric": metric,
                "best_params": study.best_params}
        with open(config.DATA / "herbert_optuna_best_params.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
        study.trials_dataframe().to_csv(config.DATA / "herbert_optuna_all_trials.csv",
                                        index=False)
    return study


def load_best_params(path=None):
    """Najlepsze hiperparametry z pliku JSON zapisanego przez tune_hyperparameters."""
    import json
    path = path or (config.DATA / "herbert_optuna_best_params.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("best_params", {})


# ----------------------------------------------------------------------------
# Trening finalny i predykcje
# ----------------------------------------------------------------------------
def train_final_model(best_params, tokenized_train, tokenized_valid, tokenizer,
                      class_weights, model_name=None, out_dir=None,
                      early_stopping_patience=2, force_retrain=True):
    """Trenuje model finalny na najlepszej konfiguracji i zapisuje go na dysk.

    Ewaluacja po kazdej epoce, wczesne zatrzymanie wybiera najlepsza epoke wedlug
    F1-macro na zbiorze walidacyjnym. Uczy wylacznie na zbiorze treningowym;
    walidacyjny sluzy do wyboru epoki. Wymaga GPU. Zwraca gotowy Trainer.

    force_retrain=False wczytuje model z dysku, jesli juz tam jest.
    """
    import os
    import shutil

    import torch
    from transformers import (AutoConfig, AutoModelForSequenceClassification,
                              EarlyStoppingCallback, TrainingArguments)

    model_name = model_name or config.HERBERT_MODEL
    out_dir = str(out_dir or config.HERBERT_DIR)
    WeightedLossTrainer = build_weighted_loss_trainer_cls()

    wsad = best_params.get("per_device_train_batch_size", 32)
    grad_accum = max(1, TARGET_BATCH_SIZE // wsad)
    frozen = best_params.get("num_frozen_layers", 0)

    args = TrainingArguments(
        output_dir=str(config.MODELS / "herbert_wyniki"),
        fp16=torch.cuda.is_available(),
        num_train_epochs=best_params.get("num_train_epochs", 4),
        per_device_train_batch_size=wsad,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        # load_best_model_at_end wymaga zapisu co epoke, ale po treningu uzywany
        # jest tylko model najlepszy. Bez limitu kazdy przebieg zostawia tyle
        # checkpointow, ile epok - przy HerBERcie okolo gigabajta kazdy.
        save_total_limit=1,
        learning_rate=best_params.get("learning_rate", 2e-5),
        weight_decay=best_params.get("weight_decay", 0.01),
        warmup_ratio=best_params.get("warmup_ratio", 0.0),
        label_smoothing_factor=best_params.get("label_smoothing_factor", 0.0),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        lr_scheduler_type=best_params.get("lr_scheduler_type", "linear"),
        adam_beta2=best_params.get("adam_beta2", 0.999),
        logging_steps=100,
        disable_tqdm=False,
    )

    gotowy = os.path.exists(os.path.join(out_dir, "config.json"))
    if not force_retrain and gotowy:
        model = AutoModelForSequenceClassification.from_pretrained(out_dir)
        freeze_model_layers(model, frozen)
        return WeightedLossTrainer(
            class_weights=class_weights,
            focal_loss_gamma=best_params.get("focal_loss_gamma"),
            decay_rate=best_params.get("decay_rate"),
            model=model, args=args,
            train_dataset=tokenized_train, eval_dataset=tokenized_valid,
            compute_metrics=compute_metrics)

    if force_retrain and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # UWAGA: zmienna nazywa sie model_cfg, a nie config - inaczej przeslonilaby
    # modul konfiguracyjny projektu (taki blad byl w notebooku 01a).
    model_cfg = AutoConfig.from_pretrained(model_name, num_labels=3)
    model_cfg.hidden_dropout_prob = best_params.get("hidden_dropout_prob", 0.1)
    model_cfg.attention_probs_dropout_prob = best_params.get("attention_probs_dropout_prob", 0.1)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_cfg)
    freeze_model_layers(model, frozen)

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        focal_loss_gamma=best_params.get("focal_loss_gamma"),
        decay_rate=best_params.get("decay_rate"),
        model=model, args=args,
        train_dataset=tokenized_train, eval_dataset=tokenized_valid,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)])

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return trainer


def predict_splits(trainer, tokenized, frames, save=True, out_csv=None):
    """Predykcje i prawdopodobienstwa dla wszystkich podzbiorow w jednym pliku.

    tokenized - slownik {"train": ds, "valid": ds, "test": ds}
    frames    - slownik {"train": df, "valid": df, "test": df} w tej samej kolejnosci wierszy

    Zapisuje data/dane_z_prawdopodobienstwem.csv - wejscie dla notebookow 01b,
    02, 03 i 04. Zwraca (polaczona ramka, slownik surowych wynikow predykcji).
    """
    import pandas as pd
    from scipy.special import softmax

    out_csv = out_csv or config.PROBS_CSV
    surowe, czesci = {}, []
    for nazwa in ("train", "valid", "test"):
        if nazwa not in tokenized:
            continue
        wynik = trainer.predict(tokenized[nazwa])
        surowe[nazwa] = wynik
        probs = softmax(wynik.predictions, axis=1)
        d = frames[nazwa].copy()
        d["predicted_label"] = wynik.predictions.argmax(-1)
        for k in range(probs.shape[1]):
            d[f"probability_topic_{k}"] = probs[:, k]
        d["split"] = nazwa
        czesci.append(d)

    dane = pd.concat(czesci, ignore_index=True)
    if "date_parsed" in dane.columns:
        dane["date_parsed"] = pd.to_datetime(dane["date_parsed"], errors="coerce")
        dane = dane.sort_values("date_parsed").reset_index(drop=True)
    if save:
        dane.to_csv(out_csv, index=False)
    return dane, surowe


# ----------------------------------------------------------------------------
# Baseline'y (CPU, bez GPU)
# ----------------------------------------------------------------------------
def frozen_embeddings(texts, model_name=None, batch_size=64, max_length=None,
                      device=None, cache_npy=None):
    """Embeddingi z NIEdostrojonego HerBERTa: srednia po tokenach z maska uwagi.

    Sluzy do punktu odniesienia "HerBERT bez dostrojenia". Enkoder
    jest zamrozony, wiec jedyna uczona czescia jest glowica liniowa doklejana
    przez wywolujacego - roznica wobec modelu dostrojonego mowi, ile wnosi samo
    dostrojenie wag enkodera.

    Usrednianie wazone maska, nie stanem tokena [CLS]: surowy BERT nie byl uczony
    tak, by [CLS] niosl reprezentacje zdania, wiec odczyt z [CLS] zanizalby
    baseline i zawyzal przewage modelu dostrojonego.

    cache_npy - sciezka pliku .npy. Policzenie embeddingow dla calego korpusu
    zajmuje kilka minut na GPU, wiec wynik zapisuje sie na dysk i przy kolejnym
    wywolaniu jest stamtad czytany, o ile liczba wierszy sie zgadza.
    """
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    teksty = [str(x) for x in texts]
    if cache_npy is not None and Path(cache_npy).is_file():
        zapisane = np.load(cache_npy)
        if zapisane.shape[0] == len(teksty):
            print(f"Embeddingi zamrozonego enkodera wczytane z {cache_npy.name}")
            return zapisane
        print(f"{cache_npy.name}: {zapisane.shape[0]} wierszy wobec {len(teksty)} "
              f"oczekiwanych - liczenie od nowa")

    model_name = model_name or config.HERBERT_MODEL
    max_length = max_length or config.MAX_LENGTH
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    out = []
    with torch.no_grad():
        for start in range(0, len(teksty), batch_size):
            partia = teksty[start:start + batch_size]
            enc = tokenizer(partia, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            stany = model(**enc).last_hidden_state              # [B, T, H]
            maska = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            srednia = (stany * maska).sum(1) / maska.sum(1).clamp(min=1e-9)
            out.append(srednia.cpu().numpy())
    emb = np.vstack(out)

    if cache_npy is not None:
        Path(cache_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_npy, emb)
        print(f"Embeddingi zamrozonego enkodera zapisane -> {cache_npy.name}")
    return emb


def baselines(train_df, test_df, valid_df=None, return_vocab=False,
              include_frozen=True, device=None):
    """Punkty odniesienia uczone na TRAIN, oceniane na VALID i TEST.

    Cztery modele, w kolejnosci tabeli punktow odniesienia: klasa wiekszosciowa, TF-IDF
    z regresja logistyczna, TF-IDF z liniowym SVM oraz HerBERT bez dostrojenia
    (zamrozony enkoder, usrednianie po tokenach, regresja logistyczna).

    valid_df=None zachowuje stare zachowanie (same kolumny dla testu). Podanie
    valid_df daje uklad kolumn tabeli punktow odniesienia: accuracy i f1_macro osobno dla
    zbioru walidacyjnego i testowego.

    return_vocab=True dolacza rozmiar slownika TF-IDF po filtrowaniu i przed nim
    (liczby podawane w README).

    include_frozen=False pomija baseline na zamrozonym enkoderze. Pierwsze jego
    policzenie wymaga przejscia calego korpusu przez HerBERTa i trwa kilka minut;
    kolejne wywolania czytaja embeddingi z data/frozen_herbert_embeddings.npy.
    """
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, f1_score

    Xtr, ytr = train_df["text"].astype(str), train_df["label"].astype(int)
    splits = {}
    if valid_df is not None:
        splits["valid"] = (valid_df["text"].astype(str), valid_df["label"].astype(int))
    splits["test"] = (test_df["text"].astype(str), test_df["label"].astype(int))

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
    Xtr_t = tfidf.fit_transform(Xtr)

    models = [
        ("Klasa większościowa", DummyClassifier(strategy="most_frequent"), False),
        ("TF-IDF + regresja logistyczna",
         LogisticRegression(max_iter=1000, class_weight="balanced"), True),
        ("TF-IDF + liniowy SVM", LinearSVC(class_weight="balanced"), True),
    ]

    rows = []
    for name, clf, use_tfidf in models:
        clf.fit(Xtr_t if use_tfidf else Xtr, ytr)
        row = {"model": name}
        for split_name, (X, y) in splits.items():
            yp = clf.predict(tfidf.transform(X) if use_tfidf else X)
            row[f"accuracy_{split_name}"] = accuracy_score(y, yp)
            row[f"f1_macro_{split_name}"] = f1_score(y, yp, average="macro")
        rows.append(row)

    if include_frozen:
        # Embeddingi liczone jednym przejsciem dla wszystkich podzbiorow naraz,
        # zeby cache na dysku obejmowal caly korpus i nie rozpadal sie na czesci.
        kolejnosc = [("train", Xtr)] + [(n, X) for n, (X, _) in splits.items()]
        wszystkie = pd.concat([X for _, X in kolejnosc], ignore_index=True)
        emb = frozen_embeddings(wszystkie.tolist(), device=device,
                                cache_npy=config.DATA / "frozen_herbert_embeddings.npy")
        granice, start = {}, 0
        for nazwa, X in kolejnosc:
            granice[nazwa] = emb[start:start + len(X)]
            start += len(X)

        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(granice["train"], ytr)
        row = {"model": "HerBERT (bez dostrojenia)"}
        for split_name, (X, y) in splits.items():
            yp = clf.predict(granice[split_name])
            row[f"accuracy_{split_name}"] = accuracy_score(y, yp)
            row[f"f1_macro_{split_name}"] = f1_score(y, yp, average="macro")
        rows.append(row)

    out = pd.DataFrame(rows)
    if valid_df is None:                       # zgodnosc wsteczna z poprzednia sygnatura
        out = out.rename(columns={"accuracy_test": "accuracy", "f1_macro_test": "f1_macro"})
    if return_vocab:
        unfiltered = TfidfVectorizer(ngram_range=(1, 2)).fit(Xtr)
        return out, {"vocab_filtered": len(tfidf.vocabulary_),
                     "vocab_unfiltered": len(unfiltered.vocabulary_)}
    return out


def prepare_datasets(frames, model_name=None, max_length=None):
    """Tokenizuje podzbiory i liczy wagi klas - wszystko, czego potrzebuje trening.

    frames - slownik {"train": df, "valid": df, "test": df} z data.load_splits.
    Zwraca slownik: tokenizer, tokenized (te same klucze co frames) oraz
    class_weights policzone WYLACZNIE na zbiorze treningowym, zeby rozklad
    walidacyjny i testowy nie wplynal na uczenie.
    """
    from datasets import Dataset
    from transformers import AutoTokenizer

    model_name = model_name or config.HERBERT_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenize = make_tokenize_function(tokenizer, max_length=max_length)

    tokenized = {nazwa: Dataset.from_pandas(d).map(tokenize, batched=True)
                 for nazwa, d in frames.items()}
    return {
        "tokenizer": tokenizer,
        "tokenized": tokenized,
        "class_weights": compute_class_weights(frames["train"]["label"].values),
    }


def loss_ablation(best_params, tokenized_train, tokenized_valid, tokenized_test,
                  class_weights, seeds=(42, 1, 2), model_name=None,
                  ablation_seed=42):
    """Ablacja funkcji straty oraz rozrzut wyniku po ziarnach losowych.

    Odpowiada na dwa pytania. Pierwsze: ile wnosi strata ogniskowa ponad sama
    wazona entropie krzyzowa - oba warianty uczone sa na tym samym ziarnie, wiec
    roznica nie pochodzi z losowej inicjalizacji. Drugie: jak duzy jest rozrzut
    wyniku miedzy ziarnami - bez tej liczby nie wiadomo, czy przewaga nad
    baseline'em miesci sie w szumie.

    Kazdy bieg to pelny fine-tuning na GPU, wiec caly eksperyment trwa
    kilkakrotnie dluzej niz sam trening finalny. Zwraca DataFrame
    [wariant, seed, f1_macro_test].
    """
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import f1_score
    from transformers import (AutoConfig, AutoModelForSequenceClassification,
                              EarlyStoppingCallback, TrainingArguments, set_seed)

    model_name = model_name or config.HERBERT_MODEL
    WeightedLossTrainer = build_weighted_loss_trainer_cls()

    def biegnij(use_focal, seed):
        set_seed(seed)
        model_cfg = AutoConfig.from_pretrained(model_name, num_labels=3)
        model_cfg.hidden_dropout_prob = best_params.get("hidden_dropout_prob", 0.1)
        model_cfg.attention_probs_dropout_prob = best_params.get(
            "attention_probs_dropout_prob", 0.1)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=model_cfg)
        freeze_model_layers(model, best_params.get("num_frozen_layers", 0))

        args = TrainingArguments(
            output_dir=str(config.MODELS / f"herbert_ablacja_seed{seed}"),
            num_train_epochs=best_params.get("num_train_epochs", 4),
            per_device_train_batch_size=best_params.get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=32,
            learning_rate=best_params.get("learning_rate", 2e-5),
            weight_decay=best_params.get("weight_decay", 0.01),
            warmup_ratio=best_params.get("warmup_ratio", 0.0),
            label_smoothing_factor=best_params.get("label_smoothing_factor", 0.0),
            lr_scheduler_type=best_params.get("lr_scheduler_type", "linear"),
            adam_beta2=best_params.get("adam_beta2", 0.999),
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model="f1_macro",
            greater_is_better=True, fp16=torch.cuda.is_available(),
            seed=seed, report_to="none", disable_tqdm=True)

        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            focal_loss_gamma=(best_params.get("focal_loss_gamma") if use_focal else None),
            decay_rate=best_params.get("decay_rate"),
            model=model, args=args,
            train_dataset=tokenized_train, eval_dataset=tokenized_valid,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
        trainer.train()
        wynik = trainer.predict(tokenized_test)
        return float(f1_score(wynik.label_ids, wynik.predictions.argmax(-1), average="macro"))

    wiersze = [
        {"wariant": "focal + wagi klas", "seed": ablation_seed,
         "f1_macro_test": biegnij(True, ablation_seed)},
        {"wariant": "sama wazona CE", "seed": ablation_seed,
         "f1_macro_test": biegnij(False, ablation_seed)},
    ]
    for seed in seeds:
        if seed == ablation_seed:
            wynik = wiersze[0]["f1_macro_test"]
        else:
            wynik = biegnij(True, seed)
        wiersze.append({"wariant": "focal + wagi klas (ziarna)", "seed": seed,
                        "f1_macro_test": wynik})

    tab = pd.DataFrame(wiersze)
    ziarna = tab[tab["wariant"] == "focal + wagi klas (ziarna)"]["f1_macro_test"]
    print(f"Po {len(ziarna)} ziarnach: {np.mean(ziarna):.4f} +/- {np.std(ziarna):.4f}")
    return tab
