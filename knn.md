# K-Nearest Neighbors (KNN)

Implements the KNN algorithm for classification and regression.

## Theory & Math
KNN predicts the label of a sample by looking at the $k$ closest points in the training set (using a distance metric, e.g., Euclidean distance) and taking a majority vote (classification) or mean (regression).

## Usage
```python
from src.knn import KNNClassifier
from src.linear_algebra import Vector

X = [Vector([0, 0]), Vector([1, 1]), Vector([2, 2])]
y = [0, 1, 1]
knn = KNNClassifier(k=3)
knn.fit(X, y)
pred = knn.predict(Vector([1.5, 1.5]))
print(pred)
```

## Classes
- `KNNClassifier(k=3)`: KNN classifier for classification tasks

See the source code for more details.
