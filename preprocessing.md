# Preprocessing

Data preprocessing utilities for ML workflows.

## Theory & Math
Preprocessing transforms raw data into a suitable format for ML. Common steps:
- Normalization
- Standardization
- Encoding categorical variables

## Usage
```python
from src.preprocessing import normalize, standardize

X = [[1, 2], [2, 3], [3, 4]]
print(normalize(X))
print(standardize(X))
```

## Functions
- `normalize(X)`
- `standardize(X)`

See the source code for more details.
