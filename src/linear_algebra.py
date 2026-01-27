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
        Docstring for __add__
        
        :param self: Objeto instanciado que chama o metodo
        :param other: Outro vetor a ser somado
        :type other: 'Vector'
        :return: Retorna a soma dos dois vetores
        :rtype: Vector
        """
        if len(self.components) != len(other.components):
            raise ValueError(f"Vetores devem ter o mesmo tamanho. {len(self.components)} != {len(other.components)}")
        else:
            result = [x + y for x,y in zip(self.components,other.components)]
            return Vector(result)
    
    def __sub__(self,other:'Vector') -> 'Vector':
        """
        Docstring for __sub__
        
        :param self: O vetor passado como objeto
        :param other: O vetor que será subtraído
        :type other: 'Vector'
        :return: Retorna o resultado da subtração entre o vetor self pelo other
        :rtype: Vector
        """
        if len(self.components) != len(other.components):
            raise ValueError(f"Vetores devem ter o mesmo tamanho. {len(self.components)} != {len(other.components)}")
        else:
            result = [x - y for x,y in zip(self.components,other.components)]
            return Vector(result)
    
    def __mul__(self,scalar:Scalar)->'Vector':
        """
        Docstring for __mul__
        
        :param self: O objeto que chama a função
        :param scalar: Um número (float ou int)
        :type scalar: Scalar (float ou int)
        :return: Retorna um novo vetor, que cada elemento é o resultado do produto entre o escalar e os elementos do vetor passado
        :rtype: Vector
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
        Docstring for sum_of_squares
        
        :param self: Vetor instanciado como objeto
        :return: Retorna o valor do produto escalar entre o vetor e ele mesmo
        :rtype: Scalar
        """
        return self.dot_product(self)
    
    def magnitude(self) -> float:
        """
        Calcula o comprimento/módulo/intensidade/magnitude de um vetor
        
        :param self: O vetor do qual o comprimento será calculado
        :return: Retorna o comprimento do vetor
        :rtype: float
        """
        return sqrt(self.sum_of_squares())
    
    def euclidian_distance(self,other:'Vector') -> float:
        """
        Calcula a distância euclidiana entre dois vetores
        
        :param self: Vetor que chamou a função
        :param other: Outro vetor
        :type other: 'Vector'
        :return: Distância euclidiana entre self e other
        :rtype: float
        """

        diff_vector = self - other

        return diff_vector.magnitude()