import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
X = np.array([
    [2, 60], [3, 65], [4, 70], [5, 75], [6, 80],
    [7, 85], [8, 90], [9, 95], [1, 55], [4, 68]
])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
model = LogisticRegression()
model.fit(X, y)
y_prob = model.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, y_prob)
auc = roc_auc_score(y, y_prob)
print("AUC Score:", auc)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
