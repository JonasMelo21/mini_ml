# Clustering

Implements the K-Means clustering algorithm for unsupervised learning.

## Theory & Math
K-Means is an unsupervised learning algorithm that partitions data into $k$ clusters by minimizing the within-cluster sum of squares. The algorithm iteratively:
1. Assigns each point to the nearest centroid
2. Recalculates centroids as the mean of assigned points
3. Repeats until convergence (or max iterations reached)

## Usage
```python
from src.clustering import KMeans
from src.linear_algebra import Vector

# Create data points
data = [
    Vector([0, 0]),
    Vector([1, 1]),
    Vector([10, 10]),
    Vector([11, 11])
]

# Fit the model with k=2 clusters
kmeans = KMeans(k=2)
kmeans.fit(data, max_iteractions=100)

# Predict cluster for a new point
cluster_id = kmeans.predict(Vector([0.5, 0.5]))
print(f"Point belongs to cluster: {cluster_id}")
```

## Classes
- `KMeans(k=3)`: K-Means clustering with $k$ clusters
  - `fit(data, max_iteractions=100)`: Train the model on data
  - `predict(point)`: Return the cluster index (0 to k-1) for a point

## Key Methods

### fit(data, max_iteractions=100)
Trains the K-Means model by finding optimal centroids.

**Parameters:**
- `data` (List[Vector]): Training data points
- `max_iteractions` (int): Maximum iterations for convergence (default: 100)

### predict(point)
Assigns a point to the nearest cluster.

**Parameters:**
- `point` (Vector): A single data point

**Returns:**
- `int`: Cluster index (0 to k-1)

See the source code for more details.
