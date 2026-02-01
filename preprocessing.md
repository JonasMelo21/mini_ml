# Preprocessing

Data preprocessing utilities for ML workflows.

## Theory & Math
Preprocessing transforms raw data into a suitable format for ML. Common steps include:
- Normalization: Scale features into a fixed range (e.g. [0, 1])
- Standardization: Transform features to have zero mean and unit variance
- Encoding categorical variables

## Current Status

At present, this module only documents common preprocessing concepts.
Programmatic utilities such as `normalize(X)` and `standardize(X)` are
planned but **not yet implemented** in `src.preprocessing`, so they cannot
currently be imported or used in code.

## Planned Functions
- `normalize(X)`: Scale features into a fixed range (e.g. [0, 1])
- `standardize(X)`: Transform features to have zero mean and unit variance

These functions will be documented here in more detail once they are
implemented in the source code.

See the source code for more details.
