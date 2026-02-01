from typing import List
from collections import Counter
from src.linear_algebra import Vector

class KNearestNeighbors:
    def __init__(self, k: int = 3):
        self.k = k
        self.train_points: List[Vector] = []
        self.train_labels: List[int] = []

    def fit(self, x: List[Vector], y: List[int]) -> None:
        # Apenas armazena os dados.
        self.train_points = x
        self.train_labels = y

    def predict(self, new_point: Vector) -> int:
        """
        Retorna a classe mais votada pelos K vizinhos mais próximos.
        """
        # 1. Calcule distâncias e combine com labels (Crie lista de tuplas)
        # Dica: Use zip(self.train_points, self.train_labels)
        # Dica: Use point.distance(new_point) que vc já criou no Vector
        distancias_labels = [ (new_point.euclidian_distance(xi),yi) for xi,yi in zip(self.train_points,self.train_labels)]
        
        # 2. Ordene a lista da menor distância para a maior
        distancias_labels_sorted = sorted(distancias_labels,key=lambda x: x[0])

        # 3. Pegue os primeiros k elementos (slice [:k])
        nearest_neighboors = distancias_labels_sorted[:self.k]

        # 4. Extraia apenas os labels desses k vizinhos
        labels = [label[1] for label in nearest_neighboors]
        
        # 5. Votação (Counter). Retorne o mais comum.
        return Counter(labels).most_common(1)[0][0]