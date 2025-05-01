# House Price Prediction Using Linear Regression (From Scratch) 🏠

This project implements Linear Regression written from scratch. It primarily focuses on house price prediction in Azerbaijan to test the model's functionality.

## Key Features 📌
- Real-world dataset from the Azerbaijani housing market 🇦🇿
- Linear Regression model built from scratch ✍
- Feature normalization and early stopping ⚙
- Cost function visualization 📉
- R² score for model evaluation 📊

## Project Directory Structure
```
📁 house-price-prediction-using-linear-regression-from-scratch
├── 📄 linear_regression.py         # Custom LinearRegression1 class
├── 📄 housePricePrediction.ipynb  # Demo: load data, predict, plot
├── 📄 real_estate_listings.csv    # Dataset (Rooms, Floor, Height, Area, Price)
├── 📄 requirements.txt            # Dependencies: numpy, pandas, matplotlib
├── 📄 .gitignore                  # Ignored files
├── 📄 README.md                   # Project documentation
├── 📄 LICENSE                     # MIT License
```

## Model Overview

### Class: `LinearRegression1`
#### Constructor:
```python
LinearRegression1(epochs=10000, alpha=0.01, normalize=True, plot_cost=False, tol=1e-9)
```
| Param     | Type    | Default | Description                        |
|-----------|---------|---------|------------------------------------|
| `epochs`  | int     | 10000   | Number of training iterations      |
| `alpha`   | float   | 0.01    | Learning rate                      |
| `normalize` | bool  | True    | Apply feature normalization        |
| `plot_cost` | bool  | False   | Plot cost over epochs if True      |
| `tol`     | float   | 1e-9    | Early stopping threshold           |

#### Methods
- `fit(X, Y)` – Train the model
- `predict(X)` – Predict target values
- `score(X, Y)` – Return R² score
- `costFunction(X, Y, w)` – Internal cost function

## 📈 Example Usage
```python
from linear_regression import LinearRegression1
import pandas as pd

# Load data
df = pd.read_csv("real_estate_listings.csv")
X = df[["Rooms", "Floor", "Height", "Area"]]
y = df["Price"]

# Train model
model = LinearRegression1(plot_cost=True)
model.fit(X, y)

# Predict
model.predict([[3, 10, 10, 150.0]])
# Output: array([343384.25867255])

# Evaluate
print("R² score:", model.score(X, y))
# Output: np.float64(0.3912458236401598)
```

## ⚙ Installation
```bash
pip install -r requirements.txt
```

## 🪪 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🧾 Credits
- **Author**: Imran Mammadov
- **University**: ADA University
- **Dataset**: Real estate listings taken from [bina.az](https://bina.az)