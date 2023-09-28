import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.impute import SimpleImputer


your_dataset = pd.read_csv("C:\\Users\\KALPANA K\\OneDrive\\dataset.csv")


your_dataset = your_dataset.drop(columns=["Unnamed: 32"])


your_dataset['diagnosis'] = your_dataset['diagnosis'].map({'M': 1, 'B': 0})

X = your_dataset.drop(columns=['diagnosis'])
y = your_dataset['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)


y_prob = model.predict_proba(X_test)[:, 1]


plt.figure(figsize=(8, 18))


plt.subplot(3, 1, 1)
plt.scatter(y_test, y_prob, alpha=0.5)
plt.xlabel("Actual Values (0: Benign, 1: Malignant)")
plt.ylabel("Predicted Probabilities")
plt.title("Scatter Plot of Actual vs. Predicted Probabilities")


feature_importance = model.feature_importances_
feature_names = your_dataset.columns[:-1]
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_feature_importance = feature_importance[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

plt.subplot(3, 1, 2)
plt.barh(sorted_feature_names, sorted_feature_importance)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")


fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.subplot(3, 1, 3)
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()
