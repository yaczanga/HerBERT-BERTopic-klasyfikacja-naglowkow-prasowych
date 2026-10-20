"""Sciezki i stale uzywane przez wszystkie notebooki i moduly src/."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Katalogi
DATA        = ROOT / "data"           # korpus i dane posrednie
MODELS      = ROOT / "models"         # modele i bazy prob Optuny (poza repozytorium)
OUTPUTS_DIR = ROOT / "outputs"        # wszystko wygenerowane: tabele i rysunki
FIGURES     = OUTPUTS_DIR / "figures" # rysunki zapisywane przez notebooki

# Brakujace katalogi powstaja przy imporcie, zeby zapis nie konczyl sie bledem
for _d in (DATA, OUTPUTS_DIR, FIGURES, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

# Pliki
RAW_CSV    = DATA / "korpus_surowy.csv"              # korpus surowy, wejscie etapu 0
TRAIN_CSV  = DATA / "train_data.csv"                 # podzbior uczacy
VALID_CSV  = DATA / "valid_data.csv"                 # podzbior walidacyjny
TEST_CSV   = DATA / "test_data.csv"                  # podzbior testowy
PROBS_CSV  = DATA / "dane_z_prawdopodobienstwem.csv" # predykcje i prawdopodobienstwa HerBERTa
SPLIT_INFO = DATA / "split_info.json"                # granice i liczebnosci podzialu

# Wyniki scraperow
OKO_CSV         = DATA / "oko_press_articles.csv" # artykuly OKO.press
# Niezalezna i RP. Plik NIE jest dolaczony do repozytorium: powstaje dopiero po
# uruchomieniu python -m src.scraping. Korpus surowy data/korpus_surowy.csv zlozono
# z niego i z OKO_CSV doraznie, krok scalajacy nie zachowal sie - patrz README,
# sekcja "Ograniczenia".
NEWSPAPERS_CSV  = DATA / "dataset.csv"

# Modele
HERBERT_DIR       = MODELS / "herbert_best_model"          # dostrojony klasyfikator
BERTOPIC_DIR      = MODELS / "best_bertopic_model"         # model tematyczny podstawowy
BERTOPIC_MERGED   = MODELS / "bertopic_model_merged"       # model tematyczny po scaleniu
BERTOPIC_REDUCED  = MODELS / "best_bertopic_model_reduced" # model po redukcji odstajacych
MISCLASSIFIED_DIR = MODELS / "misclassified_topic_model"   # model tematyczny na pomylkach

# Bazy prob Optuny (SQLite), po jednej na kazde z trzech strojen
HERBERT_OPTUNA_DB  = f"sqlite:///{(MODELS / 'herbert_optuna.db').as_posix()}"
BERTOPIC_OPTUNA_DB = f"sqlite:///{(MODELS / 'bertopic_optuna.db').as_posix()}"
MISCLASSIFIED_OPTUNA_DB = f"sqlite:///{(MODELS / 'misclassified_optuna.db').as_posix()}"

# Stale eksperymentu
RANDOM_SEED   = 42
GAZETY        = {0: "Niezależna", 1: "OKO.press", 2: "RP"}
HERBERT_MODEL = "allegro/herbert-base-cased"
MAX_LENGTH    = 128                 # maksymalna dlugosc tokenizacji naglowka
DATA_START    = "2022-01-01"        # dolna granica korpusu
VALID_YEARS   = 1                   # dlugosc okna walidacyjnego w latach
TEST_YEARS    = 1                   # dlugosc okna testowego w latach

# Polskie slowa funkcyjne odsiewane przy liczeniu spojnosci tematycznej
EXTRA_STOPWORDS = {
    "to", "sie", "nie", "na", "jest", "co", "jak", "czy", "za", "po", "do",
    "od", "dla", "te", "ta", "tym", "tego", "tej", "ten", "lub", "oraz",
    "ale", "tylko", "już", "też", "wiec", "bo", "gdy",
}
