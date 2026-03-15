import random 
from typing import List 
from src.linear_algebra import Vector, vector_mean, euclidian_distance

class KMeans:
    def __init__(self,k: int = 3):
        self.k = k 
        self.centroids = []
    
    def fit(self,data:List[Vector], max_iteractions : int = 100) -> None:
        """
        Executa o algoritmo KMeans para encontrar os centroids
        """

        # Escolhendo n vetores aleatorios para começar
        self.centroids = random.sample(data,self.k)
        
        
        # Assigning vectors to clusters
        for _ in range(max_iteractions):
            # Creating a list of clusters
            clusters = [ [] for _ in range(self.k)]
    
            # Finding the closest centroids
            for point in data:
                distances = [euclidian_distance(point,self.centroids[i]) for i in range(self.k)]
                closest_centroid_index = distances.index(min(distances))
                clusters[closest_centroid_index].append(point)

            # New centroids (center of mass) and re-assigning
            new_centroids = []
            for i,cluster in enumerate(clusters):
                if cluster:
                    new_center = vector_mean(cluster)
                    new_centroids.append(new_center)
                
                if not cluster:
                    new_centroids.append(self.centroids[i])
            
            self.centroids = new_centroids
    
    def predict(self,point:Vector) -> int:
        """
        Retorna o indice do cluster (0 a k-1) ao qual o ponto pertence
        """

        # Measuring the distance between all centroids and return the centroid its assing to
        distances = [euclidian_distance(point,self.centroids[i]) for i in range(self.k)] 
        centroid_index = distances.index(min(distances))
        return centroid_index   