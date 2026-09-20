import numpy as np
from sklearn.linear_model import LogisticRegression

# Custom student dataset
# Features: [Study Hours, Attendance]
X = np.array([
    [2, 60],
    [3, 65],
    [4, 70],
    [5, 75],
    [6, 80],
    [7, 85],
    [8, 90],
    [9, 95],
    [1, 55],
    [4, 68]
])

# Target: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])

# Create and train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Predict for a new student
study_hours = 6
attendance = 82

prediction = model.predict([[study_hours, attendance]])

if prediction[0] == 1:
    print("Result: Pass")
else:
    print("Result: Fail")