# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    f1_score,
    recall_score,
    accuracy_score,
    classification_report
)

sns.set(style='white')

# Load Data
dataset = pd.read_csv('iris.csv')

# Clean column names
dataset.columns = [
    col.strip().lower().replace(' ', '_').replace('(cm)', '').strip('_')
    for col in dataset.columns
]

# Check whether the expected columns exist
print("Dataset columns:", dataset.columns.tolist())

# Feature Engineering
dataset['sepal_length_width_ratio'] = (
    dataset['sepal_length'] / dataset['sepal_width']
)

dataset['petal_length_width_ratio'] = (
    dataset['petal_length'] / dataset['petal_width']
)

# Select features and target
dataset = dataset[
    [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'sepal_length_width_ratio',
        'petal_length_width_ratio',
        'target'
    ]
]

# Split Data
train_data, test_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=44,
    stratify=dataset['target']
)

# Separate features and target
X_train = train_data.drop('target', axis=1).values.astype('float32')
y_train = train_data['target'].values.astype('int32')

X_test = test_data.drop('target', axis=1).values.astype('float32')
y_test = test_data['target'].values.astype('int32')

# Class names
target_names = ['setosa', 'versicolor', 'virginica']

# ==========================================
# 1. Logistic Regression
# ==========================================

logreg = LogisticRegression(
    C=0.0001,
    solver='lbfgs',
    max_iter=1000
)

logreg.fit(X_train, y_train)

predictions_lr = logreg.predict(X_test)

# Evaluation metrics
cm_lr = confusion_matrix(y_test, predictions_lr)

f1_lr = f1_score(
    y_test, predictions_lr, average='macro', zero_division=0
)

prec_lr = precision_score(
    y_test, predictions_lr, average='macro', zero_division=0
)

recall_lr = recall_score(
    y_test, predictions_lr, average='macro', zero_division=0
)

train_acc_lr = accuracy_score(
    y_train, logreg.predict(X_train)
) * 100

test_acc_lr = accuracy_score(
    y_test, predictions_lr
) * 100

# ==========================================
# 2. Random Forest Classifier
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=44
)

rf_model.fit(X_train, y_train)

predictions_rf = rf_model.predict(X_test)

# Evaluation metrics
cm_rf = confusion_matrix(y_test, predictions_rf)

f1_rf = f1_score(
    y_test, predictions_rf, average='macro', zero_division=0
)

prec_rf = precision_score(
    y_test, predictions_rf, average='macro', zero_division=0
)

recall_rf = recall_score(
    y_test, predictions_rf, average='macro', zero_division=0
)

train_acc_rf = accuracy_score(
    y_train, rf_model.predict(X_train)
) * 100

test_acc_rf = accuracy_score(
    y_test, predictions_rf
) * 100

# ==========================================
# 3. Confusion Matrix Plotting Function
# ==========================================

def plot_cm(cm, target_names, title, filename, normalize=True):

    cm_display = cm.astype(float).copy()

    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)

        cm_display = np.divide(
            cm_display,
            row_sums,
            out=np.zeros_like(cm_display),
            where=row_sums != 0
        )

    accuracy = np.trace(cm) / cm.sum() if cm.sum() else 0
    misclass = 1 - accuracy

    plt.figure(figsize=(9, 6))

    sns.heatmap(
        cm_display,
        annot=True,
        fmt='.2f' if normalize else 'g',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names
    )

    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel(
        f'Predicted Label\nAccuracy={accuracy:.4f}; '
        f'Misclassification={misclass:.4f}'
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close()


# Plot Logistic Regression confusion matrix
plot_cm(
    cm_lr,
    target_names,
    title='Confusion Matrix (Logistic Regression)',
    filename='ConfusionMatrix.png',
    normalize=True
)

# Plot Random Forest confusion matrix
plot_cm(
    cm_rf,
    target_names,
    title='Confusion Matrix (Random Forest)',
    filename='ConfusionMatrix_RF.png',
    normalize=True
)

# ==========================================
# 4. Random Forest Feature Importance
# ==========================================

importances = rf_model.feature_importances_

labels = dataset.columns[:-1]

feature_df = pd.DataFrame({
    'feature': labels,
    'importance': importances
})

features = feature_df.sort_values(
    by='importance',
    ascending=False
)

plt.figure(figsize=(12, 6))

ax = sns.barplot(
    x='importance',
    y='feature',
    data=features
)

ax.set_xlabel('Importance', fontsize=14)
ax.set_ylabel('Feature', fontsize=14)
ax.set_title('Random Forest Feature Importances', fontsize=14)

plt.tight_layout()

# Save before showing
plt.savefig('FeatureImportance.png', dpi=120, bbox_inches='tight')
plt.show()
plt.close()

# ==========================================
# 5. Print Classification Reports
# ==========================================

print("\nLogistic Regression Classification Report:")
print(classification_report(
    y_test,
    predictions_lr,
    target_names=target_names,
    zero_division=0
))

print("\nRandom Forest Classification Report:")
print(classification_report(
    y_test,
    predictions_rf,
    target_names=target_names,
    zero_division=0
))

# ==========================================
# 6. Save Scores to File
# ==========================================

with open('scores.txt', 'w') as score:

    score.write("RANDOM FOREST CLASSIFIER\n")
    score.write(f"Training Accuracy: {train_acc_rf:.2f}%\n")
    score.write(f"Testing Accuracy: {test_acc_rf:.2f}%\n")
    score.write(f"F1 Score (Macro): {f1_rf:.4f}\n")
    score.write(f"Recall Score (Macro): {recall_rf:.4f}\n")
    score.write(f"Precision Score (Macro): {prec_rf:.4f}\n")

    score.write("\nLOGISTIC REGRESSION\n")
    score.write(f"Training Accuracy: {train_acc_lr:.2f}%\n")
    score.write(f"Testing Accuracy: {test_acc_lr:.2f}%\n")
    score.write(f"F1 Score (Macro): {f1_lr:.4f}\n")
    score.write(f"Recall Score (Macro): {recall_lr:.4f}\n")
    score.write(f"Precision Score (Macro): {prec_lr:.4f}\n")

print("\nModel evaluation completed successfully!")
print("Results saved to scores.txt")
print("Confusion matrices and feature importance plots saved.")