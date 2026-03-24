import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import os

# Create directory for plots
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def generate_accounting_data(n_samples=10000):
    """Génère un dataset simulé d'écritures comptables (inspiré du PCGM)."""
    
    # Paramètres normaux
    comptes = ['5141', '4111', '4411', '6111', '7111', '3421', '4455'] # PCGM typique
    users = ['Comptable_A', 'Comptable_B', 'Comptable_C', 'Sys_Auto']
    descriptions = ['Règlement client', 'Paiement fournisseur', 'Achat marchandises', 'Vente marchandises', 'Frais bancaires', 'TVA']
    
    # Génération des données de base (Normales)
    data = []
    
    for _ in range(n_samples):
        # Montant log-normal pour suivre approximativement Benford
        montant = np.round(np.random.lognormal(mean=7, sigma=1.5), 2)
        if montant < 10:
            montant += 10 # Eviter les très petits montants
            
        compte = np.random.choice(comptes, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05])
        user = np.random.choice(users, p=[0.3, 0.3, 0.1, 0.3])
        
        # Générer des dates en semaine principalement
        day = np.random.choice(range(1, 29))
        month = np.random.randint(1, 13)
        hour = int(np.random.normal(12, 3))
        hour = max(8, min(18, hour)) # Heures de bureau
        
        desc = np.random.choice(descriptions)
        
        data.append([montant, f"2023-{month:02d}-{day:02d} {hour:02d}:00:00", compte, desc, user, 0])
        
    df = pd.DataFrame(data, columns=['Montant', 'Date', 'Compte', 'Description', 'Utilisateur', 'ecriture_suspecte'])
    
    # Injection d'anomalies (Fraude / Erreurs) ~ 5%
    n_anomalies = int(n_samples * 0.05)
    anomalies_indices = np.random.choice(df.index, n_anomalies, replace=False)
    
    for idx in anomalies_indices:
        df.loc[idx, 'ecriture_suspecte'] = 1
        type_anomalie = np.random.randint(1, 4)
        
        if type_anomalie == 1:
            # Anomalie 1: Montant rond très élevé par un utilisateur inhabituel hors heures
            df.loc[idx, 'Montant'] = np.random.choice([50000.0, 100000.0, 75000.0, 99999.0])
            df.loc[idx, 'Utilisateur'] = 'Directeur_Fin'
            df.loc[idx, 'Date'] = f"2023-12-31 23:59:00" # Fin d'année tard
            df.loc[idx, 'Description'] = 'Ajustement manuel'
            
        elif type_anomalie == 2:
            # Anomalie 2: Ecriture passée le week-end
            df.loc[idx, 'Date'] = f"2023-05-14 03:15:00" # Dimanche matin
            df.loc[idx, 'Montant'] = np.round(np.random.uniform(10000, 50000), 2)
            
        elif type_anomalie == 3:
            # Anomalie 3 : Fractionnement (Smurfing), montants juste en dessous d'un seuil
            df.loc[idx, 'Montant'] = np.random.uniform(9900, 9999)
            df.loc[idx, 'Compte'] = '6111'
            df.loc[idx, 'Utilisateur'] = 'Comptable_A'
            df.loc[idx, 'Description'] = 'Frais divers'

    df['Date'] = pd.to_datetime(df['Date'])
    
    # Feature Engineering
    df['Heure'] = df['Date'].dt.hour
    df['JourSemaine'] = df['Date'].dt.dayofweek
    df['Est_Weekend'] = df['JourSemaine'].apply(lambda x: 1 if x >= 5 else 0)
    df['Premier_Chiffre'] = df['Montant'].astype(str).str[0].astype(int)
    
    return df

def perform_eda(df):
    print("--- Analyse Exploratoire ---")
    print(df.head())
    print("\nRépartition de la cible:")
    print(df['ecriture_suspecte'].value_counts())
    
    # 1. Distribution des montants
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Montant', hue='ecriture_suspecte', bins=50, log_scale=True)
    plt.title("Distribution des Montants (échelle logarithmique)")
    plt.savefig(f'{output_dir}/eda_montants_hist.png')
    plt.close()
    
    # 2. Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='ecriture_suspecte', y='Montant')
    plt.yscale('log')
    plt.title("Boxplot des montants par statut d'anomalie")
    plt.savefig(f'{output_dir}/eda_montants_box.png')
    plt.close()
    
    # 3. Heatmap de corrélation
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Heatmap des corrélations")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_correlation.png')
    plt.close()

def analyze_benford(df):
    print("\n--- Loi de Benford ---")
    # Calcul des fréquences réelles
    proportions_reelles = df['Premier_Chiffre'].value_counts(normalize=True).sort_index()
    
    # Loi théorique de Benford
    chiffres = np.arange(1, 10)
    proportions_theoriques = np.log10(1 + 1/chiffres)
    
    # Tracé
    plt.figure(figsize=(10, 6))
    plt.bar(chiffres - 0.2, proportions_reelles.reindex(chiffres, fill_value=0), width=0.4, label='Données Réelles', color='blue')
    plt.bar(chiffres + 0.2, proportions_theoriques, width=0.4, label='Loi de Benford', color='orange', alpha=0.7)
    plt.plot(chiffres, proportions_theoriques, color='red', marker='o')
    plt.xticks(chiffres)
    plt.xlabel('Premier Chiffre')
    plt.ylabel('Proportion')
    plt.title('Analyse de la Loi de Benford')
    plt.legend()
    plt.savefig(f'{output_dir}/benford_analysis.png')
    plt.close()

def machine_learning(df):
    print("\n--- Machine Learning ---")
    # Encodage des variables catégorielles (ex: variables factices / dummies)
    df_encoded = pd.get_dummies(df, columns=['Compte', 'Utilisateur', 'Description'], drop_first=True)
    
    features = [col for col in df_encoded.columns if col not in ['Date', 'ecriture_suspecte']]
    X = df_encoded[features]
    y = df_encoded['ecriture_suspecte']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    
    print("\nRandom Forest Results:")
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print(f"Precision: {precision_score(y_test, rf_preds):.4f}")
    print(f"Recall: {recall_score(y_test, rf_preds):.4f}")
    
    # Matrice de confusion RF
    cm = confusion_matrix(y_test, rf_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion - Random Forest')
    plt.ylabel('Vrai labels')
    plt.xlabel('Prédictions')
    plt.savefig(f'{output_dir}/ml_rf_confusion.png')
    plt.close()
    
    # Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10] # Top 10
    plt.figure(figsize=(10, 6))
    plt.title("Importance des Variables (Random Forest - Top 10)")
    plt.bar(range(10), importances[indices], align="center")
    plt.xticks(range(10), [features[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ml_rf_importance.png')
    plt.close()
    
    # 2. Isolation Forest (Non supervisé - on simule qu'on n'a pas les labels pour l'entraînement)
    # L'Isolation Forest prédit -1 pour les anomalies et 1 pour les normales
    iso_f = IsolationForest(contamination=0.05, random_state=42)
    iso_f.fit(X) # Entraînement sur tout le dataset
    iso_preds = iso_f.predict(X)
    iso_preds_binary = [1 if p == -1 else 0 for p in iso_preds]
    
    print("\nIsolation Forest Results (Global dataset):")
    print(f"Accuracy vs Real labels: {accuracy_score(y, iso_preds_binary):.4f}")
    print(f"Precision: {precision_score(y, iso_preds_binary):.4f}")
    print(f"Recall: {recall_score(y, iso_preds_binary):.4f}")
    
    # Visualisation des anomalies (PCA pour projeter en 2D pour IF)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # StandardScaler avant PCA pour un meilleur résultat
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iso_preds_binary, cmap='coolwarm', alpha=0.6)
    plt.title('Anomalies détectées par Isolation Forest (Projection PCA)')
    plt.colorbar(sc, label='0: Normal, 1: Anomalie')
    plt.savefig(f'{output_dir}/ml_if_anomalies.png')
    plt.close()
    
    # 3. Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    
    # Scaling requis pour LR sinon convergence difficile
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)
    
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, lr_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbe ROC - Logistic Regression')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/ml_lr_roc.png')
    plt.close()
    
    # 4. K-Means (Optionnel Avancé)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    kmeans_preds = kmeans.predict(X_scaled)
    
    # Les clusters peuvent être inversés par rapport aux labels (0 peut être anomalie ou inverse)
    # Cherchons quel cluster correspond le plus aux anomalies
    if np.mean(y[kmeans_preds == 1]) > np.mean(y[kmeans_preds == 0]):
        cluster_anomalies = 1
    else:
        cluster_anomalies = 0
        
    kmeans_anomalies = [1 if p == cluster_anomalies else 0 for p in kmeans_preds]
    
    print("\nK-Means Clustering Results:")
    print(f"Accuracy vs Real labels: {accuracy_score(y, kmeans_anomalies):.4f}")
    
if __name__ == "__main__":
    print("Génération des données...")
    df = generate_accounting_data(10000)
    
    # Sauvegarde du dataset pour référence
    df.to_csv('journal_entries.csv', index=False)
    
    perform_eda(df)
    analyze_benford(df)
    machine_learning(df)
    print("\nTerminé. Les graphiques ont été sauvegardés dans le dossier 'plots'.")
