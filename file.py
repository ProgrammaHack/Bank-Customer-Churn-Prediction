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

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

os.makedirs("output_grafici_reali", exist_ok=True)

MODEL_PATH = "models/rf_model.pkl"
DATA_PATH = "data/churn.csv"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Errore: Non ho trovato il modello in '{MODEL_PATH}'.")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Errore: Non ho trovato il file dei dati in '{DATA_PATH}'.")

model = joblib.load(MODEL_PATH)

df = load_data(DATA_PATH)
df = clean_data(df)

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
plt.savefig('output_grafici_reali/0_distribuzione_target.png', dpi=300)
plt.close()

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
plt.savefig('output_grafici_reali/1_feature_importance_reale.png', dpi=300)
plt.close()

y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= 0.35).astype(int)

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
plt.savefig('output_grafici_reali/2_curva_roc_reale.png', dpi=300)
plt.close()

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
plt.savefig('output_grafici_reali/3_matrice_confusione_reale.png', dpi=300)
plt.close()

if 'Age' in X_test.columns and 'Balance' in X_test.columns:
    plt.figure(figsize=(10, 6))
    df_cluster = X_test[['Age', 'Balance']].copy()
    df_cluster['Exited'] = y_test.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster[['Age', 'Balance']])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)

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

    sns.scatterplot(
        x='Age',
        y='Balance',
        hue='Cluster_Nome',
        style='Exited',
        data=df_cluster,
        palette='Set2',
        alpha=0.6,
        s=60
    )
    plt.title('Segmentazione Strategica Clienti e Pattern di Churn')
    plt.xlabel('Età (Age)')
    plt.ylabel('Saldo Conto (Balance)')
    plt.legend(title='Segmenti & Esito (x = Abbandono)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('output_grafici_reali/4_clustering_clienti_reali.png', dpi=300)
    plt.close()
