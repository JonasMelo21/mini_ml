# mini_ml

A minimalistic, educational machine learning library implemented from scratch in Python. This project is designed for learning and experimentation, providing clear, well-documented code for core ML algorithms and utilities.

## Features
- Core ML algorithms: KNN, Linear Regression, Logistic Regression, and more
- Utility modules: Activation functions, optimization, preprocessing, statistics, and linear algebra
- Simple, readable code for educational purposes
- Comprehensive tests for each module

## Installation

### Linux (Recommended)
```bash
git clone https://github.com/YOUR_USERNAME/mini_ml.git
cd mini_ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows
- Use [Git Bash](https://gitforwindows.org/) or [WSL](https://docs.microsoft.com/en-us/windows/wsl/) for best compatibility.
- Follow the Linux instructions above inside your terminal.

## Usage
Each module in `src/` has its own documentation and usage guide:
- [Activation Functions](activation.md)
- [KNN](knn.md)
- [Linear Algebra](linear_algebra.md)
- [Linear Model](linear_model.md)
- [Logistic Model](logistic_model.md)
- [Optimization](optimization.md)
- [Preprocessing](preprocessing.md)
- [Statistics](statistics.md)

Each guide covers:
- Theory, math, and intuition
- Main functions/classes to use
- Short code examples

## Example: Using KNN
```python
from src.knn import KNNClassifier

X_train = [[0, 0], [1, 1], [2, 2]]
y_train = [0, 1, 1]

knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)
pred = knn.predict([[1.5, 1.5]])
print(pred)
```

## Contributing
1. Fork the repository and create a new branch from `main`.
2. Add or improve features. **Write tests** for your code in the `tests/` folder, following the existing style.
3. Run all tests with:
   ```bash
   pytest
   ```
   Ensure all tests pass. Add a screenshot of your terminal showing all tests passing to your PR.
4. Use [semantic commit messages](https://www.conventionalcommits.org/en/v1.0.0/).
5. Open a pull request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

This project is for educational purposes. Contributions and feedback are welcome!
