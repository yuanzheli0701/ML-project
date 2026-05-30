
"""Generate confusion matrix and final evaluation plots"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Load data
df = pd.read_csv(r'C:\Users\13968\Documents\ML机器学习\features.csv')
rare_mask = df['bug type'].isin(['Bee & Bumblebee', 'Dragonfly'])
df = df[~rare_mask].copy()

feature_cols = [c for c in df.columns if c not in ['ID', 'bug type', 'species']]
X = df[feature_cols].values.astype(np.float64)
y = df['bug type'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Best model: SVM with C=10, gamma='scale'
model = SVC(C=10, gamma='scale', kernel='rbf', random_state=42)

# Cross-validated predictions
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(model, X_scaled, y_encoded, cv=cv)

# Confusion matrix
cm = confusion_matrix(y_encoded, y_pred_cv)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(cmap='Blues', ax=ax, values_format='d')
ax.set_title('Confusion Matrix - SVM (5-fold CV)')
plt.tight_layout()
plt.savefig(r'C:\Users\13968\Documents\ML机器学习\fig_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print('Confusion matrix saved!')

# Also do a combined summary plot
from sklearn.metrics import classification_report
report = classification_report(y_encoded, y_pred_cv, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()

fig, ax = plt.subplots(figsize=(12, 6))
metrics = report_df.loc[le.classes_, ['precision', 'recall', 'f1-score']]
metrics.plot(kind='bar', ax=ax)
ax.set_title('Per-Class Performance - SVM (5-fold CV)')
ax.set_ylabel('Score')
ax.set_xlabel('Bug Type')
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\13968\Documents\ML机器学习\fig_per_class.png', dpi=150, bbox_inches='tight')
plt.close()

print('Per-class performance plot saved!')
print('\nClassification Report:')
print(classification_report(y_encoded, y_pred_cv, target_names=le.classes_))
