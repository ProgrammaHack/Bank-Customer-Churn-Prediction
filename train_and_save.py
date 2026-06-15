import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.preprocessing import load_data, clean_data
from src.model import train_model
from src.evaluation import evaluate


# =========================
# CARICAMENTO E PULIZIA DATI
# =========================
df = load_data("data/churn.csv")
df = clean_data(df)

# =========================
# FEATURES E TARGET
# =========================
X = df.drop("Exited", axis=1)
y = df["Exited"]

# =========================
# TRAIN / TEST SPLIT
# =========================
# stratify=y mantiene la stessa proporzione delle classi
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# TRAIN MODELLO
# =========================
model = train_model(X_train, y_train)

# =========================
# VALUTAZIONE BASE
# =========================
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)

# =========================
# SALVATAGGIO MODELLO
# =========================
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/rf_model.pkl")

print("\nModello salvato in models/rf_model.pkl")