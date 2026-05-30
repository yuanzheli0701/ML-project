
"""Visualization and ML Training for ML Project: To Bee or Not To Bee"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score, adjusted_rand_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

WORK_DIR = r'C:\Users\13968\Documents\ML机器学习'

def main():
    print("="*60)
    print("ML Project: Visualization and Model Training")
    print("="*60)
    
    # Load data
    df = pd.read_csv(r'C:\Users\13968\Documents\ML机器学习\features.csv')
    print(f'\nLoaded {len(df)} samples, {len(df.columns)-3} features')
    
    # Prepare features and labels
    # Remove classes with only 1 sample (they can't be split for CV)
    rare_mask = df['bug type'].isin(['Bee & Bumblebee', 'Dragonfly'])
    df_filtered = df[~rare_mask].copy()
    print(f'Removed {rare_mask.sum()} rare-class samples (Bee & Bumblebee, Dragonfly)')
    
    feature_cols = [c for c in df.columns if c not in ['ID', 'bug type', 'species']]
    X = df_filtered[feature_cols].values
    y = df_filtered['bug type'].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f'\nBug type distribution:')
    for label in np.unique(y):
        count = (y == label).sum()
        print(f'  {label}: {count}')
    
    # ============ VISUALIZATION ============
    print('\n' + '='*60)
    print('Creating visualizations...')
    
    # 1. Bug type distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    type_counts = df['bug type'].value_counts()
    axes[0].bar(type_counts.index, type_counts.values, color=sns.color_palette('Set2', len(type_counts)))
    axes[0].set_title('Bug Type Distribution')
    axes[0].set_xlabel('Bug Type')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    species_counts = df['species'].value_counts().head(10)
    axes[1].barh(species_counts.index[::-1], species_counts.values[::-1], color=sns.color_palette('Set3', 10))
    axes[1].set_title('Top 10 Species Distribution')
    axes[1].set_xlabel('Count')
    plt.tight_layout()
    plt.savefig(r'C:\Users\13968\Documents\ML机器学习\fig_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  [1/5] Distribution plots saved')
    
    # 2. PCA Projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 8))
    for label in np.unique(y):
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.7, s=30)
    ax.set_title(f'PCA Projection (EV: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(r'C:\Users\13968\Documents\ML机器学习\fig_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  [2/5] PCA projection saved')
    
    # 3. t-SNE Projection
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 8))
    for label in np.unique(y):
        mask = y == label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=label, alpha=0.7, s=30)
    ax.set_title('t-SNE Projection')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(r'C:\Users\13968\Documents\ML机器学习\fig_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  [3/5] t-SNE projection saved')
    
    # 4. UMAP Projection
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 8))
    for label in np.unique(y):
        mask = y == label
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], label=label, alpha=0.7, s=30)
    ax.set_title('UMAP Projection')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(r'C:\Users\13968\Documents\ML机器学习\fig_umap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  [4/5] UMAP projection saved')
    
    # 5. Feature correlation heatmap
    fig, ax = plt.subplots(figsize=(16, 12))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, cmap='RdBu_r', center=0, ax=ax, xticklabels=True, yticklabels=True)
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(r'C:\Users\13968\Documents\ML机器学习\fig_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  [5/5] Correlation heatmap saved')
    
    # ============ MODEL TRAINING ============
    print('\n' + '='*60)
    print('Training ML Models...')
    print('='*60)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    # 1. Supervised: Logistic Regression (non-DL, non-ensemble)
    print('\n[1] Logistic Regression...')
    lr = LogisticRegression(max_iter=2000, multi_class='multinomial')
    lr_scores = cross_val_score(lr, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    results.append({'Model': 'Logistic Regression', 'CV Mean': lr_scores.mean(), 'CV Std': lr_scores.std()})
    print(f'    5-fold CV Accuracy: {lr_scores.mean():.4f} +/- {lr_scores.std():.4f}')
    
    # 2. Supervised: KNN (non-DL, non-ensemble)
    print('\n[2] K-Nearest Neighbors...')
    knn = KNeighborsClassifier(n_neighbors=5)
    knn_scores = cross_val_score(knn, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    results.append({'Model': 'KNN (k=5)', 'CV Mean': knn_scores.mean(), 'CV Std': knn_scores.std()})
    print(f'    5-fold CV Accuracy: {knn_scores.mean():.4f} +/- {knn_scores.std():.4f}')
    
    # 3. Supervised: SVC (non-DL, non-ensemble) - third method for comparison
    print('\n[3] SVM (RBF Kernel)...')
    svm = SVC(kernel='rbf', random_state=42)
    svm_scores = cross_val_score(svm, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    results.append({'Model': 'SVM (RBF)', 'CV Mean': svm_scores.mean(), 'CV Std': svm_scores.std()})
    print(f'    5-fold CV Accuracy: {svm_scores.mean():.4f} +/- {svm_scores.std():.4f}')
    
    # 4. Ensemble: Random Forest
    print('\n[4] Random Forest (Ensemble)...')
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_scores = cross_val_score(rf, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    results.append({'Model': 'Random Forest', 'CV Mean': rf_scores.mean(), 'CV Std': rf_scores.std()})
    print(f'    5-fold CV Accuracy: {rf_scores.mean():.4f} +/- {rf_scores.std():.4f}')
    
    # 5. Clustering: KMeans
    print('\n[5] KMeans Clustering...')
    n_clusters = len(np.unique(y_encoded))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    sil_kmeans = silhouette_score(X_scaled, kmeans_labels)
    ari_kmeans = adjusted_rand_score(y_encoded, kmeans_labels)
    results.append({'Model': 'KMeans', 'CV Mean': sil_kmeans, 'CV Std': ari_kmeans})
    print(f'    Silhouette Score: {sil_kmeans:.4f}')
    print(f'    Adjusted Rand Index: {ari_kmeans:.4f}')
    
    # 6. Clustering: DBSCAN
    print('\n[6] DBSCAN Clustering...')
    dbscan = DBSCAN(eps=2.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    if n_clusters_db > 1:
        mask_valid = dbscan_labels != -1
        if mask_valid.sum() > 1 and len(set(dbscan_labels[mask_valid])) > 1:
            sil_dbscan = silhouette_score(X_scaled[mask_valid], dbscan_labels[mask_valid])
            ari_dbscan = adjusted_rand_score(y_encoded[mask_valid], dbscan_labels[mask_valid])
        else:
            sil_dbscan, ari_dbscan = -1, -1
    else:
        sil_dbscan, ari_dbscan = -1, -1
    results.append({'Model': 'DBSCAN', 'CV Mean': sil_dbscan, 'CV Std': ari_dbscan})
    print(f'    Clusters found: {n_clusters_db}')
    print(f'    Silhouette Score: {sil_dbscan:.4f}')
    print(f'    Adjusted Rand Index: {ari_dbscan:.4f}')
    
    # 7. LDA (Linear Discriminant Analysis) - bonus method
    print('\n[7] LDA...')
    lda = LinearDiscriminantAnalysis()
    lda_scores = cross_val_score(lda, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    results.append({'Model': 'LDA', 'CV Mean': lda_scores.mean(), 'CV Std': lda_scores.std()})
    print(f'    5-fold CV Accuracy: {lda_scores.mean():.4f} +/- {lda_scores.std():.4f}')
    
    # Results summary
    results_df = pd.DataFrame(results)
    print('\n' + '='*60)
    print('MODEL COMPARISON')
    print('='*60)
    print(results_df.to_string(index=False))
    results_df.to_csv(r'C:\Users\13968\Documents\ML机器学习\model_results.csv', index=False)
    
    # Find best model for final prediction
    supervised_models = {
        'Logistic Regression': lr,
        'KNN': knn,
        'SVM': svm,
        'Random Forest': rf,
        'LDA': lda
    }
    
    # Detailed evaluation with train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print('\n' + '='*60)
    print('DETAILED EVALUATION (80/20 split)')
    print('='*60)
    
    best_model_name = None
    best_accuracy = 0
    
    for name, model in supervised_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
        print(f'\n--- {name} ---')
        print(f'Accuracy: {acc:.4f}')
        print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print(f'\nBest model: {best_model_name} (Accuracy: {best_accuracy:.4f})')
    
    # Train final best model on all data and save
    print('\nTraining final best model on all training data...')
    best_model = supervised_models[best_model_name]
    best_model.fit(X_scaled, y_encoded)
    
    import pickle
    with open(r'C:\Users\13968\Documents\ML机器学习\best_model.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler, 'label_encoder': le}, f)
    print(f'Best model ({best_model_name}) saved!')
    
    print('\nDone! All visualizations and models saved.')
    return best_model_name, best_accuracy

if __name__ == '__main__':
    best_name, best_acc = main()
