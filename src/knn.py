from typing import List
from collections import Counter
from src.linear_algebra import Vector

class KNNClassifier:
    def __init__(self, k: int = 3):
        self.k = k
        self.train_points: List[Vector] = []
        self.train_labels: List[int] = []

    def fit(self, X: List[Vector], y: List[int]) -> None:
        """
        Stores the training data for KNN.
        Args:
            X (List[Vector]): Training feature vectors.
            y (List[int]): Training labels.
        """
        self.train_points = X
        self.train_labels = y

    def predict(self, new_point: Vector) -> int:
        """
        Returns the most common class among the k nearest neighbors.
        Args:
            new_point (Vector): The point to classify.
        Returns:
            int: Predicted class label.
        """
        # 1. Calculate distances and pair with labels
        distances_labels = [(new_point.euclidian_distance(xi), yi) for xi, yi in zip(self.train_points, self.train_labels)]
        # 2. Sort by distance
        distances_labels_sorted = sorted(distances_labels, key=lambda x: x[0])
        # 3. Take k nearest neighbors
        nearest_neighbors = distances_labels_sorted[:self.k]
        # 4. Extract labels
        labels = [label[1] for label in nearest_neighbors]
        # 5. Majority vote
        return Counter(labels).most_common(1)[0][0]