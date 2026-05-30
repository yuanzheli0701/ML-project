
"""Improved Model Training with Hyperparameter Tuning + Optional DL"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, silhouette_score, adjusted_rand_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

WORK_DIR = r'C:\Users\13968\Documents\ML机器学习'

def main():
    print("="*60)
    print("IMPROVED ML Training")
    print("="*60)
    
    # Load and filter
    df = pd.read_csv(r'C:\Users\13968\Documents\ML机器学习\features.csv')
    rare_mask = df['bug type'].isin(['Bee & Bumblebee', 'Dragonfly'])
    df_filtered = df[~rare_mask].copy()
    
    feature_cols = [c for c in df.columns if c not in ['ID', 'bug type', 'species']]
    X = df_filtered[feature_cols].values.astype(np.float64)
    y = df_filtered['bug type'].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # ========== GRID SEARCH FOR BEST MODELS ==========
    print('\n--- Hyperparameter Tuning ---')
    
    # RF Grid Search
    print('\n[1] Random Forest Grid Search...')
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    rf_gs = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    rf_gs.fit(X_scaled, y_encoded)
    print(f'    Best params: {rf_gs.best_params_}')
    print(f'    Best CV score: {rf_gs.best_score_:.4f}')
    
    # SVM Grid Search
    print('\n[2] SVM Grid Search...')
    svm_params = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'class_weight': ['balanced', None]
    }
    svm_gs = GridSearchCV(SVC(kernel='rbf', random_state=42), svm_params, cv=3, scoring='accuracy', n_jobs=-1)
    svm_gs.fit(X_scaled, y_encoded)
    print(f'    Best params: {svm_gs.best_params_}')
    print(f'    Best CV score: {svm_gs.best_score_:.4f}')
    
    # KNN Grid Search
    print('\n[3] KNN Grid Search...')
    knn_params = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}
    knn_gs = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3, scoring='accuracy')
    knn_gs.fit(X_scaled, y_encoded)
    print(f'    Best params: {knn_gs.best_params_}')
    print(f'    Best CV score: {knn_gs.best_score_:.4f}')
    
    # ========== FINAL EVALUATION ==========
    print('\n' + '='*60)
    print('FINAL EVALUATION WITH BEST MODELS')
    print('='*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    models = {
        'Logistic Regression (balanced)': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'KNN (best)': knn_gs.best_estimator_,
        'SVM (best)': svm_gs.best_estimator_,
        'Random Forest (best)': rf_gs.best_estimator_,
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'LDA': LinearDiscriminantAnalysis()
    }
    
    results = []
    best_model = None
    best_acc = 0
    best_name = ''
    
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({'Model': name, 'CV Mean': scores.mean(), 'CV Std': scores.std(), 'Test Acc': acc})
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
        print(f'\n--- {name} ---')
        print(f'CV Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}')
        print(f'Test Accuracy: {acc:.4f}')
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    
    results_df = pd.DataFrame(results)
    print('\n' + '='*60)
    print('FINAL RESULTS')
    print('='*60)
    print(results_df.to_string(index=False))
    results_df.to_csv(r'C:\Users\13968\Documents\ML机器学习\final_results.csv', index=False)
    
    # ========== CLUSTERING ==========
    print('\n' + '='*60)
    print('CLUSTERING RESULTS')
    print('='*60)
    n_clusters = len(np.unique(y_encoded))
    
    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    print(f'KMeans - Silhouette: {silhouette_score(X_scaled, km_labels):.4f}, ARI: {adjusted_rand_score(y_encoded, km_labels):.4f}')
    
    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(X_scaled)
    print(f'Agglomerative - Silhouette: {silhouette_score(X_scaled, agg_labels):.4f}, ARI: {adjusted_rand_score(y_encoded, agg_labels):.4f}')
    
    # DBSCAN
    db = DBSCAN(eps=2.5, min_samples=5)
    db_labels = db.fit_predict(X_scaled)
    n_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    print(f'DBSCAN - Clusters: {n_db}, Noise: {(db_labels==-1).sum()}')
    valid = db_labels != -1
    if valid.sum() > 1 and len(set(db_labels[valid])) > 1:
        print(f'DBSCAN - Silhouette: {silhouette_score(X_scaled[valid], db_labels[valid]):.4f}, ARI: {adjusted_rand_score(y_encoded[valid], db_labels[valid]):.4f}')
    
    # ========== OPTIONAL: SIMPLE DL ==========
    print('\n' + '='*60)
    print('OPTIONAL: Simple Neural Network')
    print('='*60)
    try:
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
        mlp_scores = cross_val_score(mlp, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        mlp.fit(X_train, y_train)
        mlp_pred = mlp.predict(X_test)
        mlp_acc = accuracy_score(y_test, mlp_pred)
        print(f'MLP - CV Accuracy: {mlp_scores.mean():.4f} +/- {mlp_scores.std():.4f}')
        print(f'MLP - Test Accuracy: {mlp_acc:.4f}')
        print(classification_report(y_test, mlp_pred, target_names=le.classes_, zero_division=0))
        if mlp_acc > best_acc:
            best_model = mlp
            best_name = 'MLP Neural Network'
            best_acc = mlp_acc
    except Exception as e:
        print(f'MLP failed: {e}')
    
    # ========== SAVE BEST MODEL ==========
    print(f'\nBest model: {best_name} (Test Accuracy: {best_acc:.4f})')
    best_model.fit(X_scaled, y_encoded)
    
    import pickle
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_cols': feature_cols,
        'model_name': best_name
    }
    with open(r'C:\Users\13968\Documents\ML机器学习\best_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f'Best model saved: best_model.pkl')
    
    # ========== FEATURE IMPORTANCE (for RF) ==========
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[-15:]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_cols[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top 15 Feature Importances ({best_name})')
        plt.tight_layout()
        plt.savefig(r'C:\Users\13968\Documents\ML机器学习\fig_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('Feature importance plot saved!')
    
    print('\nAll done!')

if __name__ == '__main__':
    main()
