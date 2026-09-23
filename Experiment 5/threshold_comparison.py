import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
X = np.array([
    [2, 60], [3, 65], [4, 70], [5, 75], [6, 80],
    [7, 85], [8, 90], [9, 95], [1, 55], [4, 68],
    [6, 78], [7, 82], [3, 62], [8, 88], [5, 73]
])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1])
model = LogisticRegression()
model.fit(X, y)
probabilities = model.predict_proba(X)[:, 1]
thresholds = [0.3, 0.5, 0.7]
print("--- Threshold Comparison ---")
for threshold in thresholds:
    y_pred = (probabilities >= threshold).astype(int)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
