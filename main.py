# =========================
# IMPORT FUNZIONI
# =========================
from src.preprocessing import load_data, clean_data
from src.model import train_model, predict, predict_proba
from src.evaluation import evaluate
from src.visualization import plot_roc

from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# =========================
# 1. CARICAMENTO DEL DATASET
# =========================
df = load_data("data/churn.csv")

# =========================
# 2. PULIZIA DEI DATI
# =========================
df = clean_data(df)

# =========================
# 3. SEPARAZIONE FEATURES / TARGET
# =========================
# X contiene le variabili di input
# y contiene il target da predire: Exited
X = df.drop("Exited", axis=1)
y = df["Exited"]

# =========================
# 4. TRAIN / TEST SPLIT
# =========================
# stratify=y mantiene la proporzione delle classi nel train e nel test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5. TRAIN DEL MODELLO
# =========================
model = train_model(X_train, y_train)

# =========================
# 6. PREDIZIONE
# =========================
# threshold più basso = più sensibilità verso la classe 1
threshold = 0.35
y_pred = predict(model, X_test, threshold=threshold)

# probabilità della classe 1, necessarie per la ROC
y_proba = predict_proba(model, X_test)

# =========================
# 7. VALUTAZIONE
# =========================
evaluate(y_test, y_pred)

# =========================
# 8. ROC CURVE
# =========================
plot_roc(y_test, y_proba)

# =========================
# 9. SALVATAGGIO DEL MODELLO
# =========================
# creo la cartella models se non esiste
Path("models").mkdir(exist_ok=True)

# salvo il modello allenato
joblib.dump(model, "models/rf_model.pkl")

print("\nModello salvato in models/rf_model.pkl")