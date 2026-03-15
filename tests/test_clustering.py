from src.clustering import KMeans
from src.linear_algebra import Vector, vector_mean
def test_kmeans_obvious_clusters():
    cluster_a = [
        Vector([0,0]),
        Vector([0,1]),
        Vector([1,0])
        ]
    
    cluster_b = [
        Vector([10,10]),
        Vector([10,11]),
        Vector([11,11])
        ]
    
    cluster_c = [
        Vector([20, 20]),
        Vector([20, 21]),
        Vector([21, 21])
    ]

    data = cluster_a + cluster_b + cluster_c
    model = KMeans(3)
    model.fit(data)

    pred_a1 = model.predict(Vector([0.5,0.5]))
    pred_a2 = model.predict(Vector([0,0.5]))

    assert pred_a1 == pred_a2
    
    pred_b1 = model.predict(Vector([10.5,10.5]))
    pred_b2 = model.predict(Vector([10,10.5]))
    assert pred_b1 == pred_b2 