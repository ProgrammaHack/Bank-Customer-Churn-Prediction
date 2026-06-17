import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.preprocessing import load_data, clean_data

# Impostazioni di stile per i grafici
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Nome della cartella di output
OUTPUT_DIR = "output_grafici_reali_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "models/rf_model.pkl"
DATA_PATH = "data/churn.csv"

# Verifica esistenza file
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Errore: Non ho trovato il modello in '{MODEL_PATH}'.")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Errore: Non ho trovato il file dei dati in '{DATA_PATH}'.")

# Caricamento modello e dati
model = joblib.load(MODEL_PATH)
df = load_data(DATA_PATH)
df = clean_data(df)

# ==========================================
# 0. GRAFICO: DISTRIBUZIONE TARGET
# ==========================================
plt.figure(figsize=(6, 4))
churn_counts = df["Exited"].value_counts()
churn_pct = df["Exited"].value_counts(normalize=True) * 100

ax_dist = sns.barplot(
    x=churn_counts.index,
    y=churn_counts.values,
    hue=churn_counts.index,
    palette=['#2ecc71', '#e74c3c'],
    legend=False
)

for i, p in enumerate(ax_dist.patches):
    count = churn_counts.values[i]
    pct = churn_pct.values[i]
    ax_dist.annotate(f"{count}\n({pct:.1f}%)",
                     (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=12)

plt.title("Distribuzione del Target: Sbilanciamento delle Classi")
plt.xlabel("Stato del Cliente")
plt.ylabel("Numero di Clienti")
ax_dist.set_xticklabels(['Rimasti (0)', 'Abbandoni (1)'])
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/0_distribuzione_target.png', dpi=300)
plt.close()

# Split delle feature e target per i grafici successivi
X = df.drop("Exited", axis=1)
y = df["Exited"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

if isinstance(y_test, pd.DataFrame):
    y_test = y_test.iloc[:, 0]

# ==========================================
# 1. GRAFICO: FEATURE IMPORTANCE
# ==========================================
plt.figure(figsize=(9, 5))
features = model.feature_names_in_
importances_values = model.feature_importances_

df_imp = pd.DataFrame({'Variable': features, 'Importance': importances_values})
df_imp = df_imp.sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Variable', data=df_imp, hue='Variable', palette='viridis', legend=False)
plt.xlabel('Importanza Relativa (Gini Importance)')
plt.ylabel('Variabili estratte dal tuo Modello')
plt.title('Quali fattori guidano le decisioni del Modello Reale?')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/1_feature_importance_reale.png', dpi=300)
plt.close()

# Predizioni per metriche di performance
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= 0.35).astype(int)

# ==========================================
# 2. GRAFICO: CURVA ROC
# ==========================================
plt.figure(figsize=(7, 6))
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'Curva ROC Reale (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificatore Casuale')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasso di Falsi Positivi (1 - Specificità)')
plt.ylabel('Tasso di Veri Positivi (Sensibilità / Recall)')
plt.title('Curva ROC - Performance del tuo Modello')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/2_curva_roc_reale.png', dpi=300)
plt.close()

# ==========================================
# 3. GRAFICO: MATRICE DI CONFUSIONE
# ==========================================
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
labels = [f"{v}\n({p:.1%})" for v, p in zip(cm.flatten(), cm_percent.flatten())]
labels = np.asarray(labels).reshape(2, 2)

sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False,
            xticklabels=['Rimane (0)', 'Abbandona (1)'],
            yticklabels=['Rimane (0)', 'Abbandona (1)'])
plt.xlabel('Previsione Modello (Soglia 0.35)')
plt.ylabel('Valore Reale Effettivo')
plt.title('Matrice di Confusione Reale del Modello')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/3_matrice_confusione_reale.png', dpi=300)
plt.close()

# ==========================================
# 4. GRAFICO: CLUSTERING (OTTIMIZZATO)
# ==========================================
if 'Age' in X_test.columns and 'Balance' in X_test.columns:
    plt.figure(figsize=(10, 6))
    df_cluster = X_test[['Age', 'Balance']].copy()
    df_cluster['Exited'] = y_test.values

    # Standardizzazione e calcolo dei cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster[['Age', 'Balance']])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)

    # Ridenominazione dei cluster in base ai centri numerici
    cluster_centers = df_cluster.groupby('Cluster')[['Age', 'Balance']].mean()
    senior_cluster = cluster_centers['Age'].idxmax()
    remaining_clusters = [c for c in cluster_centers.index if c != senior_cluster]

    if cluster_centers.loc[remaining_clusters[0], 'Balance'] > cluster_centers.loc[remaining_clusters[1], 'Balance']:
        rich_cluster = remaining_clusters[0]
        poor_cluster = remaining_clusters[1]
    else:
        rich_cluster = remaining_clusters[1]
        poor_cluster = remaining_clusters[0]

    cluster_names = {
        senior_cluster: "Clienti Senior (Over 45)",
        rich_cluster: "Clienti Giovani/Adulti - Saldo Medio-Alto",
        poor_cluster: "Clienti Giovani - Saldo Basso/Zero"
    }
    df_cluster['Cluster_Nome'] = df_cluster['Cluster'].map(cluster_names)

    # Disegno dei cluster (sfondo)
    sns.scatterplot(
        x='Age',
        y='Balance',
        hue='Cluster_Nome',
        data=df_cluster,
        palette='Set2',
        alpha=0.7,
        s=55,
        marker='o',
        edgecolor='none',
        legend='brief'
    )

    # Sovrapposizione "X" nere per gli abbandoni
    sns.scatterplot(
        x='Age',
        y='Balance',
        data=df_cluster[df_cluster['Exited'] == 1],
        color='black',
        alpha=0.45,
        s=45,
        marker='X',
        label='Clienti Abbandonati (Exited = 1)'
    )

    plt.title('Segmentazione Strategica Clienti e Pattern di Churn')
    plt.xlabel('Età (Age)')
    plt.ylabel('Saldo Conto (Balance)')
    plt.legend(title='Segmenti & Esito', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/4_clustering_clienti_reali.png', dpi=300)
    plt.close()

# ==========================================
# 5. GRAFICO: CHURN RATE PER FASCE D'ETÀ (Feature 1: Age)
# ==========================================
if 'Age' in df.columns:
    plt.figure(figsize=(9, 5))

    # Creazione delle fasce d'età d'impatto aziendale
    bins = [0, 30, 40, 50, 60, 100]
    labels_age = ['<30 anni', '30-40 anni', '40-50 anni', '50-60 anni', 'Over 60']
    df['Fascia_Eta'] = pd.cut(df['Age'], bins=bins, labels=labels_age)

    # Calcolo della percentuale di churn per fascia
    age_churn = df.groupby('Fascia_Eta', observed=False)['Exited'].mean() * 100

    ax_age = sns.barplot(x=age_churn.index, y=age_churn.values, hue=age_churn.index, palette='magma', legend=False)

    for p in ax_age.patches:
        ax_age.annotate(f"{p.get_height():.1f}%",
                        (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                        ha='center', va='center', fontweight='bold', color='black', fontsize=11)

    plt.title("Impatto dell'Età sul Churn: Tasso di Abbandono per Fascia")
    plt.xlabel("Fascia d'Età dei Clienti")
    plt.ylabel("Tasso di Abbandono (%)")
    plt.ylim(0, max(age_churn.values) + 10)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/5_churn_rate_eta.png', dpi=300)
    plt.close()

# ==========================================
# 6. GRAFICO: CHURN RATE PER NUMERO DI PRODOTTI (Feature 2: NumOfProducts)
# ==========================================
if 'NumOfProducts' in df.columns:
    plt.figure(figsize=(8, 5))

    # Calcolo della percentuale di churn per numero di prodotti
    prod_churn = df.groupby('NumOfProducts')['Exited'].mean() * 100

    ax_prod = sns.barplot(x=prod_churn.index, y=prod_churn.values, hue=prod_churn.index, palette='coolwarm',
                          legend=False)

    for p in ax_prod.patches:
        ax_prod.annotate(f"{p.get_height():.1f}%",
                         (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                         ha='center', va='center', fontweight='bold', color='black', fontsize=11)

    plt.title("Anomalia dell'Offerta: Tasso di Churn per Numero di Prodotti Posseduti")
    plt.xlabel("Numero di Prodotti Attivi")
    plt.ylabel("Tasso di Churn (%)")
    plt.ylim(0, max(prod_churn.values) + 10)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/6_churn_rate_prodotti.png', dpi=300)
    plt.close()

# ==========================================
# 7. GRAFICO: COMPOSIZIONE FASCE ETÀ ALL'INTERNO DEL CHURN PER PRODOTTO
# ==========================================
if 'NumOfProducts' in df.columns and 'Fascia_Eta' in df.columns:
    plt.figure(figsize=(10, 6))

    # Filtriamo solo i clienti CHE HANNO ABBANDONATO (Exited == 1)
    df_churn_only = df[df['Exited'] == 1]

    # Calcoliamo la tabella a incrocio (cross-tab) normalizzata per colonna (prodotto)
    ct = pd.crosstab(df_churn_only['Fascia_Eta'], df_churn_only['NumOfProducts'], normalize='columns') * 100

    # Riordiniamo le fasce d'età in modo coerente
    ct = ct.reindex(labels_age)

    # Disegniamo il grafico a barre sovrapposte (stacked bar chart)
    ax_stack = ct.T.plot(kind='bar', stacked=True, cmap='viridis', edgecolor='white', width=0.6, ax=plt.gca())

    # Inseriamo i testi delle percentuali all'interno dei segmenti delle barre
    for rects in ax_stack.containers:
        labels = [f'{v:.1f}%' if v > 3 else '' for v in rects.datavalues]
        ax_stack.bar_label(rects, labels=labels, label_type='center', fontweight='bold', color='white', fontsize=10)

    plt.title("Composizione Demografica dei Clienti Persi (Churn) per Numero di Prodotti", fontsize=15)
    plt.xlabel("Numero di Prodotti Posseduti (al momento del Churn)", fontsize=13)
    plt.ylabel("Distribuzione Fasce d'Età (%)", fontsize=13)
    plt.xticks(rotation=0)
    plt.ylim(0, 105)
    plt.legend(title="Fasce d'Età", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/7_composizione_eta_nel_churn_prodotti.png', dpi=300)
    plt.close()

print(f"\nTutti gli 8 grafici salvati correttamente in: {OUTPUT_DIR}/")
