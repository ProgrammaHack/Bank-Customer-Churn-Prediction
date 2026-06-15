import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import pandas as pd

# =========================
# ROC CURVE
# =========================
def plot_roc(y_test, y_proba):
    """
    Disegna la ROC curve.
    - y_test: valori veri del target
    - y_proba: probabilità previste della classe 1
    """

    # calcolo FPR e TPR
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    # calcolo area sotto la curva
    roc_auc = auc(fpr, tpr)

    # grafico
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random model")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Random Forest")
    plt.legend()
    plt.grid(True)
    plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================
def plot_feature_importance(model, X):
    """
    Mostra quali variabili influenzano di più la decisione del modello
    """

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(x="Importance", y="Feature", data=importance)
    plt.title("Feature Importance (Random Forest)")
    plt.show()