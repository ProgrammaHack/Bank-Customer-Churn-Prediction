import pandas as pd

# =========================
# CARICAMENTO DEL DATASET
# =========================
def load_data(path):
    """
    Legge il file CSV e lo restituisce come DataFrame.
    """
    df = pd.read_csv(path)
    return df


# =========================
# PULIZIA DEI DATI
# =========================
def clean_data(df):
    """
    Rimuove colonne non utili al modello.
    In questo dataset, Surname non aiuta la predizione del churn.
    """

    # Se la colonna esiste, viene eliminata.
    # errors='ignore' evita errori se la colonna non c'è.
    df = df.drop(["Surname"], axis=1, errors="ignore")

    return df