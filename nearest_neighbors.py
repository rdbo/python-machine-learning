# Data:
# Body Fat %, Eats Chocolate
#
# Objective:
# Predict if a person eats chocolate based on their BF%

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# training dataframe
df_train = np.array([
    # BF%, Eats Chocolate
    [35.0, 1],
    [19.2, 1],
    [14.1, 0],
    [12.6, 0],
    [11.4, 0],
    [15.5, 1],
    [15.2, 0],
    [16.7, 1],
    [19.3, 1],
    [6.9,  0]
])

print("[*] Training Dataframe: " + str(df_train))

# testing dataframe
df_test = np.array([
    # BF%, Eats Chocolate
    [12.9, 0],
    [36.2, 1],
    [11.7, 0],
    [15.4, 0], # odd one out (bad data)
    [15.7, 1],
    [11.2, 0],
    [10.7, 0],
    [6.2,  0],
    [53.4, 1],
    [34.2, 1]
])

print("[*] Testing Dataframe: " + str(df_test))

# extract features and labels by getting just a specific column from
# the sets. this is done by transposition, and then getting a row
X = df_train.T[0].reshape(-1, 1) # features (reshaped into a 2D array)
print("[*] Training Features: " + str(X))
y = df_train.T[1] # labels (reshaped into a 2D array)
print("[*] Training Labels: " + str(y))

X_test = df_test.T[0].reshape(-1, 1)
y_test = df_test.T[1]

# predict
neighbors = KNeighborsClassifier(n_neighbors=3).fit(X, y) # the arrays have to be 2D
predictions = neighbors.predict(X_test)
correct_predictions = 0
for i in range(len(predictions)):
    message = f"[*] Prediction: BF {X_test[i][0]}% \t->\t{predictions[i]}"
    if predictions[i] == y_test[i]:
        correct_predictions += 1
        message += " | Correct"
    else:
        message += f" | Expected: {y_test[i]}"
    print(message)

accuracy = (correct_predictions / len(predictions)) * 100
print(f"[*] Accuracy: {accuracy}%")
