from typing import List,Union 
from math import sqrt
Scalar = Union[int,float]


class Vector:
    def __init__(self,components:List[Scalar]) -> None:
        self.components = components 
    
    def __repr__(self) -> str:
        return f"Vector({self.components})"
    
    def __add__(self, other: 'Vector') -> 'Vector':
        """
        Soma dois vetores.

        Args:
            other (Vector): Outro vetor a ser somado.

        Returns:
            Vector: Um novo vetor, resultado da soma dos dois vetores.
        """
        if len(self.components) != len(other.components):
            raise ValueError(f"Vetores devem ter o mesmo tamanho. {len(self.components)} != {len(other.components)}")
        else:
            result = [x + y for x,y in zip(self.components,other.components)]
            return Vector(result)
    
    def __sub__(self,other:'Vector') -> 'Vector':
        """
        Subtrai um vetor de outro.

        Args:
            other (Vector): O vetor que será subtraído.

        Returns:
            Vector: Um novo vetor, resultado da subtração.
        """
        if len(self.components) != len(other.components):
            raise ValueError(f"Vetores devem ter o mesmo tamanho. {len(self.components)} != {len(other.components)}")
        else:
            result = [x - y for x,y in zip(self.components,other.components)]
            return Vector(result)
    
    def __mul__(self,scalar:Scalar)->'Vector':
        """
        Multiplica o vetor por um escalar.

        Args:
            scalar (Scalar): Um número (float ou int) para multiplicar cada elemento do vetor.

        Returns:
            Vector: Um novo vetor, resultado do produto entre o escalar e o vetor.
        """
        result = [scalar * x for x in self.components]
        return Vector(result)
    
    def dot_product(self,other:'Vector') -> Scalar:
        """
        Calcula o produto escalar entre dois vetores

        Args:
            other (Vector): o outro vetor para multiplicar

        Returns:
            Scalar: Um número, resultado do produto escalar (float ou int)
        """
        if len(self.components) != len(other.components):
            raise ValueError(f"Vetores devem ter o mesmo tamanho. {len(self.components)} != {len(other.components)}")
        else:
            product = (x*y for x,y in zip(self.components,other.components))
            return sum(product)
    
    def sum_of_squares(self)->Scalar:
        """
        Calcula a soma dos quadrados dos elementos do vetor.

        Returns:
            Scalar: O valor do produto escalar do vetor por ele mesmo.
        """
        return self.dot_product(self)
    
    def magnitude(self) -> float:
        """
        Calcula o comprimento/módulo/intensidade/magnitude de um vetor.

        Returns:
            float: O comprimento (magnitude) do vetor.
        """
        return sqrt(self.sum_of_squares())
    
    def euclidian_distance(self,other:'Vector') -> float:
        """
        Calcula a distância euclidiana entre dois vetores.

        Args:
            other (Vector): Outro vetor para calcular a distância.

        Returns:
            float: Distância euclidiana entre self e other.
        """
        diff_vector = self - other
        return diff_vector.magnitude()