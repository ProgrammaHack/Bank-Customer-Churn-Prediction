import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from src.preprocessing import clean_data

# =========================
# CARICAMENTO MODELLO
# =========================
MODEL_PATH = Path("models/rf_model.pkl")
model = joblib.load(MODEL_PATH)

# =========================
# CONFIG PAGINA
# =========================
st.set_page_config(
    page_title="Bank Churn Prediction",
    layout="wide"
)

st.title("Bank Customer Churn Prediction")

st.write(
    "Inserisci i dati del cliente per prevedere se rischia di abbandonare la banca."
)

# =========================
# INPUT UTENTE (SIDEBAR)
# =========================
st.sidebar.header("Dati cliente")

# Variabili numeriche e standard
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 100, 40)
tenure = st.sidebar.slider("Tenure (anni)", 0, 10, 5)
balance = st.sidebar.slider("Balance", 0.0, 250000.0, 50000.0, step=1000.0)
num_products = st.sidebar.slider("Num of Products", 1, 4, 1)
has_cr_card = st.sidebar.selectbox("Has Credit Card", [0, 1], index=1)
is_active_member = st.sidebar.selectbox("Is Active Member", [0, 1], index=1)
estimated_salary = st.sidebar.slider("Estimated Salary", 0.0, 200000.0, 50000.0, step=1000.0)

# Nuovi input per gestire le feature mancanti (Genere e Nazione)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"], index=1)
geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"], index=0)

threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.35, 0.01)

# =========================
# PREVISIONE
# =========================
def predict_customer():
    """
    Crea un DataFrame con i dati inseriti dall’utente,
    applica il preprocessing e allinea le feature con il modello.
    """

    # 1. Costruzione del dizionario iniziale con i dati della UI
    raw_data = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }

    # 2. Creazione manuale delle feature ingegnerizzate richieste dal modello
    raw_data["Age_Tenure_product"] = age * tenure
    raw_data["Bal_sal"] = balance / (estimated_salary + 1) # +1 evita la divisione per zero
    raw_data["Cred_Bal_Sal"] = credit_score * raw_data["Bal_sal"]

    # 3. Creazione del DataFrame di partenza
    input_data = pd.DataFrame([raw_data])

    # 4. Gestione delle variabili categoriche (One-Hot Encoding)
    # Impostiamo a 1 la colonna corrispondente alla scelta dell'utente
    if gender == "Female":
        input_data["Female"] = 1
    else:
        input_data["Female"] = 0

    if geography == "France":
        input_data["France"] = 1
    elif geography == "Germany":
        input_data["Germany"] = 1
    elif geography == "Spain":
        input_data["Spain"] = 1

    # 5. Applica la funzione di pulizia del progetto (es. rimozione Surname)
    processed_input = clean_data(input_data)

    # 6. Allineamento di sicurezza con le colonne del modello (crea a 0 le colonne assenti)
    for col in model.feature_names_in_:
        if col not in processed_input.columns:
            processed_input[col] = 0

    # 7. Ordinamento identico a quello visto durante il .fit()
    final_input = processed_input[model.feature_names_in_]

    # 8. Calcolo della probabilità e predizione finale
    proba = model.predict_proba(final_input)[0, 1]
    prediction = 1 if proba >= threshold else 0

    return prediction, proba, final_input


# =========================
# LAYOUT PRINCIPALE
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input cliente")
    st.write("Modifica i valori nella sidebar per simulare un cliente reale.")
    st.info("Premi il bottone per ottenere la previsione")

with col2:
    st.subheader("Risultato modello")

    if st.button("Prevedi churn"):

        prediction, proba, final_input = predict_customer()

        st.metric("Probabilità churn", f"{proba:.2%}")
        st.progress(float(proba))

        if prediction == 1:
            st.error("Il cliente è a rischio abbandono")
        else:
            st.success("Il cliente probabilmente resterà")

        st.write("Dati elaborati inviati al modello:")
        st.dataframe(final_input)

    else:
        st.write("In attesa di input...")
