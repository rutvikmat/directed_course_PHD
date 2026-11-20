 
# ============================================================
# EXERCISE 1.1: KNN CLASSIFIER ON SHUFFLED MNIST DATA
# Book: Machine Learning - A Probabilistic Perspective
# Objective: Verify that feature permutation does not change
# the misclassification rate of a 1-NN classifier.
# ============================================================


# STEP 1: IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


print("=" * 60)
print("EXERCISE 1.1 - KNN ON SHUFFLED MNIST")
print("=" * 60)


# STEP 2: DOWNLOAD MNIST DATASET

print("\nDownloading MNIST dataset...")

mnist = fetch_openml(
    name="mnist_784",
    version=1,
    as_frame=False,
    parser="liac-arff"
)

X = mnist.data
y = mnist.target.astype(np.int64)

print("Dataset downloaded successfully!")
print("Feature matrix:", X.shape)
print("Target matrix:", y.shape)


# STEP 3: SPLIT TRAINING AND TESTING DATA

X_train = X[:60000]
y_train = y[:60000]

X_test = X[60000:]
y_test = y[60000:]

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# STEP 4: VISUALIZE MNIST DIGITS

plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    image = X_train[i].reshape(28, 28)

    plt.imshow(image, cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.suptitle("MNIST Handwritten Digits")
plt.tight_layout()
plt.show()


# STEP 5: SELECT FIRST 1000 TEST SAMPLES

X_test_1000 = X_test[:1000]
y_test_1000 = y_test[:1000]


# STEP 6: TRAIN ORIGINAL 1-NN CLASSIFIER

print("\nTraining original 1-NN classifier...")

knn = KNeighborsClassifier(
    n_neighbors=1,
    metric="euclidean",
    algorithm="brute",
    n_jobs=-1
)

knn.fit(X_train, y_train)


# STEP 7: PREDICT ORIGINAL TEST DATA

y_pred = knn.predict(X_test_1000)

accuracy = accuracy_score(y_test_1000, y_pred)

error_rate = (1 - accuracy) * 100

print("\n--- ORIGINAL MNIST RESULTS ---")

print("Correct predictions:",
      np.sum(y_pred == y_test_1000))

print("Incorrect predictions:",
      np.sum(y_pred != y_test_1000))

print("Accuracy:", accuracy * 100, "%")

print("Misclassification rate:", error_rate, "%")


# STEP 8: RANDOMLY PERMUTE THE 784 FEATURES

print("\nShuffling feature columns...")

np.random.seed(42)

n_features = X_train.shape[1]

permutation = np.random.permutation(n_features)

print("Total features:", n_features)

print("First 20 shuffled indices:", permutation[:20])


# STEP 9: APPLY SAME PERMUTATION TO TRAINING AND TEST DATA

X_train_shuffled = X_train[:, permutation]

X_test_shuffled = X_test_1000[:, permutation]

print("\nShuffled training data:",
      X_train_shuffled.shape)

print("Shuffled testing data:",
      X_test_shuffled.shape)


# STEP 10: TRAIN SHUFFLED 1-NN CLASSIFIER

print("\nTraining shuffled 1-NN classifier...")

knn_shuffled = KNeighborsClassifier(
    n_neighbors=1,
    metric="euclidean",
    algorithm="brute",
    n_jobs=-1
)

knn_shuffled.fit(X_train_shuffled, y_train)


# STEP 11: PREDICT SHUFFLED TEST DATA

y_pred_shuffled = knn_shuffled.predict(X_test_shuffled)

accuracy_shuffled = accuracy_score(
    y_test_1000,
    y_pred_shuffled
)

error_shuffled = (1 - accuracy_shuffled) * 100

print("\n--- SHUFFLED MNIST RESULTS ---")

print("Correct predictions:",
      np.sum(y_pred_shuffled == y_test_1000))

print("Incorrect predictions:",
      np.sum(y_pred_shuffled != y_test_1000))

print("Accuracy:", accuracy_shuffled * 100, "%")

print("Misclassification rate:", error_shuffled, "%")


# STEP 12: COMPARE BOTH EXPERIMENTS

results = pd.DataFrame({
    "Experiment": [
        "Original MNIST",
        "Shuffled MNIST"
    ],

    "Accuracy (%)": [
        accuracy * 100,
        accuracy_shuffled * 100
    ],

    "Error Rate (%)": [
        error_rate,
        error_shuffled
    ]
})

print("\n" + "=" * 60)
print("FINAL EXPERIMENT COMPARISON")
print("=" * 60)

print(results.to_string(index=False))


# STEP 13: VERIFY PREDICTIONS

identical = np.array_equal(
    y_pred,
    y_pred_shuffled
)

print("\nAre predictions identical?", identical)


# STEP 14: FINAL CONCLUSION

if identical:

    print("\nSUCCESS!")

    print(
        "Feature permutation did not change "
        "the KNN classification results."
    )

    print(
        "Euclidean distance is invariant "
        "under a common feature permutation."
    )

else:

    print("\nCheck the implementation and dataset.")

print("\nExercise 1.1 execution completed.")
