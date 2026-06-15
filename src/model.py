from sklearn.ensemble import RandomForestClassifier

# =========================
# TRAIN DEL MODELLO
# =========================
def train_model(X_train, y_train):
    """
    Allena una Random Forest.
    Usiamo class_weight='balanced' per dare più importanza alla classe minoritaria
    (Exited = 1, cioè i clienti che abbandonano).
    """

    model = RandomForestClassifier(
        n_estimators=300,       # numero di alberi
        random_state=42,        # risultati riproducibili
        class_weight="balanced",# utile per dataset sbilanciati
        max_depth=10,           # limita la profondità degli alberi
        min_samples_leaf=5      # evita foglie troppo piccole
    )

    model.fit(X_train, y_train)
    return model


# =========================
# PREDIZIONE CON SOGLIA
# =========================
def predict(model, X_test, threshold=0.35):
    """
    Restituisce la classe finale.
    Abbassiamo la soglia rispetto a 0.5 per aumentare il recall della classe 1.

    threshold = 0.35 significa:
    se la probabilità di churn è >= 35%, il cliente viene classificato come 1.
    """

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred


# =========================
# PREDIZIONE DELLA PROBABILITÀ
# =========================
def predict_proba(model, X_test):
    """
    Restituisce la probabilità che il cliente lasci la banca.
    Serve per ROC curve e analisi più dettagliate.
    """
    return model.predict_proba(X_test)[:, 1]