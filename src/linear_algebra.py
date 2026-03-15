from typing import List,Union 
from math import sqrt
from src.statistics import mean 
from collections import Counter 

Scalar = Union[int,float]


class Vector:
    def __init__(self,components:List[Scalar]) -> None:
        self.components = components 
    
    def __repr__(self) -> str:
        return f"Vector({self.components})"
    
    def __add__(self, other: 'Vector') -> 'Vector':
        """
        Adds two vectors.
        Args:
            other (Vector): Another vector to add.
        Returns:
            Vector: A new vector, result of the sum.
        Raises:
            ValueError: If the vectors are not the same length.
        """
        if len(self.components) != len(other.components):
            raise ValueError(f"Vectors must have the same length. {len(self.components)} != {len(other.components)}")
        else:
            result = [x + y for x, y in zip(self.components, other.components)]
            return Vector(result)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        """
        Subtracts one vector from another.
        Args:
            other (Vector): The vector to subtract.
        Returns:
            Vector: A new vector, result of the subtraction.
        Raises:
            ValueError: If the vectors are not the same length.
        """
        if len(self.components) != len(other.components):
            raise ValueError(f"Vectors must have the same length. {len(self.components)} != {len(other.components)}")
        else:
            result = [x - y for x, y in zip(self.components, other.components)]
            return Vector(result)
    
    def __mul__(self,scalar:Scalar)->'Vector':
        """
        Multiplies the vector by a scalar.

        Args:
            scalar (Scalar): A number (float or int) to multiply each element of the vector.

        Returns:
            Vector: A new vector, result of the scalar multiplication.
        """
        result = [scalar * x for x in self.components]
        return Vector(result)
    
    def dot_product(self,other:'Vector') -> Scalar:
        """
        Calculates the dot product between two vectors.

        Args:
            other (Vector): The other vector to multiply.

        Returns:
            Scalar: The result of the dot product (float or int).

        Raises:
            ValueError: If the vectors are not the same length.
        """
        if len(self.components) != len(other.components):
            raise ValueError(f"Vetores devem ter o mesmo tamanho. {len(self.components)} != {len(other.components)}")
        else:
            product = (x*y for x,y in zip(self.components,other.components))
            return sum(product)
    
    def sum_of_squares(self)->Scalar:
        """
        Calculates the sum of the squares of the vector elements.

        Returns:
            Scalar: The dot product of the vector with itself.
        """
        return self.dot_product(self)
    
    def magnitude(self) -> float:
        """
        Calculates the magnitude (length) of the vector.

        Returns:
            float: The magnitude of the vector.
        """
        return sqrt(self.sum_of_squares())
    
    def euclidian_distance(self,other:'Vector') -> float:
        """
        Calculates the Euclidean distance between two vectors.

        Args:
            other (Vector): Another vector to calculate the distance to.

        Returns:
            float: The Euclidean distance between self and other.
        """
        diff_vector = self - other
        return diff_vector.magnitude()


def vector_mean(vectors: List[Vector]) -> Vector:
    """
    Calculate the centroid of a list of vectors.
    Ex: [Vector([0,0]),Vector([4,4])] -> Vector([2,2])
    """
    # Validate if the list of Vectors is not empty
    if not vectors:
        raise ValueError("Empty Vector List")
    
    # Count how many dimensions each vector has
    dimension = [len(vector.components) for vector in vectors]
    dimension_set = set(dimension)

    if len(dimension_set) != 1:
        raise ValueError("Vectors are not the same dimension")
    
    # For each dimension i:
    # Sum the i component of ALL vectors and divide by the number of vectors
    matrix = [v.components for v in vectors]
    num_vectors = len(vectors)
    mean_components = [sum(coluna) / num_vectors for coluna in zip(*matrix)]
    return Vector(mean_components)


def euclidian_distance(v1: Vector, v2: Vector) -> float:
    """
    Calculates the Euclidean distance between two vectors.
    
    Args:
        v1 (Vector): First vector
        v2 (Vector): Second vector
    
    Returns:
        float: The Euclidean distance between v1 and v2
    """
    return v1.euclidian_distance(v2)