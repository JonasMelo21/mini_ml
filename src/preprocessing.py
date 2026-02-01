from typing import List
from .linear_algebra import Vector
from .statistics import mean, std_deviation

class StandardScaler:
    def __init__(self):
        self.means = []
        self.stds = []
    
    def fit(self,data:List[Vector]):
        matrix = [v.components for v in data]
        columns = list(zip(*matrix))

        self.means = [mean(col) for col in columns]
        self.stds = [std_deviation(col) for col in columns]
    
    def transform(self,data:List[Vector]) -> List[Vector]:
        if self.means == []:
            raise ValueError(f"Empty means list."+"\n{self.means}"+"Did you forgot to self.fit ?")
        normalized_vectores = []
        
        for vector in data:
            new_components = []

            for value,mu,std in zip(vector.components,self.means,self.stds):
                if std == 0:
                    new_val = value - mu 
                else:
                    new_val = (value - mu) / std 
                new_components.append(new_val)
            normalized_vectores.append(Vector(new_components))
        return normalized_vectores
    
    def fit_transform(self,data:List[Vector]):
        self.fit(data)
        return self.transform(data)