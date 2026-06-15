from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# VALUTAZIONE DEL MODELLO
# =========================
def evaluate(y_test, y_pred):
    """
    Stampa le metriche principali del modello.
    """

    print("\n====================")
    print("RISULTATI MODELLO")
    print("====================\n")

    # Accuratezza generale
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Precision, Recall, F1-score
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Matrice di confusione
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))