from typing import List 
from math import sqrt

def mean(data: List[float]) -> float:
    """
    Calculates the mean of a list of values.

    Args:
        data (List[float]): List of values.

    Returns:
        float: The mean of the values.

    Raises:
        ValueError: If the list is empty.
    """
    if len(data) == 0:
        raise ValueError(f"Empty list")
    else:
        return sum(data) / len(data) 

def median(data: List) -> float:
    """
    Calculates the median of a list of values.

    Args:
        data (List): List of values.

    Returns:
        float: The median value.

    Raises:
        ValueError: If the list is empty.
    """
    if not data:
        raise ValueError(f"Empty List")

    data_sorted = sorted(data)
    medium_element = len(data_sorted) // 2
    if len(data_sorted) % 2 == 0:
        return (data_sorted[medium_element - 1] + data_sorted[medium_element]) / 2
    else:
        return data_sorted[medium_element]

def variance(data:List) -> float:
    """
    Calculates the sample variance.

    Args:
        data (List): Sample data.

    Returns:
        float: The variance of the sample.
    """
    mu = mean(data)
    square_deviation = sum([(x - mu)**2 for x in data])
    return square_deviation / (len(data) - 1)

def std_deviation(data:List) -> float:
    """
    Calculates the sample standard deviation.

    Args:
        data (List): Sample data.

    Returns:
        float: The standard deviation of the sample.
    """
    return sqrt(variance(data))

def _de_mean(data:List) -> List[float]:
    """
    Subtracts the mean from each element in the sample.

    Args:
        data (List): Sample data.

    Returns:
        List[float]: List of deviations from the mean.
    """
    mu = mean(data)
    return [x - mu for x in data]

def covariance(sampleA:List,sampleB:List) -> float:
    """
    Calculates the covariance between two samples.

    Args:
        sampleA (List): First sample.
        sampleB (List): Second sample.

    Returns:
        float: The covariance between the samples.

    Raises:
        ValueError: If the samples are not the same length.
    """
    if len(sampleB) != len(sampleA):
        raise ValueError(f"Samples should be same length")
    return sum([x*y for x,y in zip(_de_mean(sampleA),_de_mean(sampleB))]) / (len(sampleB)-1)

def correlation(sampleA:List,sampleB:List) -> float:
    """
    Computes the correlation between two samples.

    Args:
        sampleA (List): First sample.
        sampleB (List): Second sample.

    Returns:
        float: The computed correlation (between -1 and 1).
    """
    if std_deviation(sampleA) * std_deviation(sampleB) == 0:    # if one of the samples has 0 standard deviation, the correlation is 0
        return 0.0
    return covariance(sampleA,sampleB) / (std_deviation(sampleA) * std_deviation(sampleB))