# Multiple Linear Regression from Scratch using NumPy (Car Fuel Economy Prediction)

A simple implementation of **Multiple Linear Regression** built completely from scratch using **NumPy** and **Gradient Descent**. This project predicts a vehicle's **fuel economy (Miles Per Gallon - MPG)** based on three vehicle characteristics: **weight**, **engine displacement**, and **horsepower**.

The purpose of this project is to understand the mathematics behind multiple linear regression without using machine learning frameworks such as scikit-learn. Every step of the learning process, including prediction, gradient computation, weight updates, and optimization, is implemented manually.

---

## Dataset

The model is trained using three vehicle features to predict fuel economy.

| Feature | Description |
|---------|-------------|
| Weight | Vehicle weight (thousand pounds) |
| Displacement | Engine displacement (cubic inches) |
| Horsepower | Engine power (HP) |
| MPG | Fuel economy (Miles Per Gallon) |

Example dataset:

| Weight | Displacement | Horsepower | MPG |
|---------|--------------|------------|-----|
| 2.3 | 140 | 90 | 36 |
| 2.8 | 160 | 110 | 33 |
| 3.1 | 180 | 120 | 31 |
| 3.5 | 220 | 150 | 28 |
| 3.8 | 250 | 170 | 25 |
| 4.0 | 280 | 190 | 23 |
| 4.3 | 300 | 210 | 21 |
| 4.6 | 330 | 230 | 19 |

---

## Mathematical Model

Unlike simple linear regression,

```
y = mx + b
```

multiple linear regression learns one weight for every input feature.

```
y = w₁x₁ + w₂x₂ + w₃x₃ + b
```

Where:

| Variable | Meaning |
|----------|---------|
| x₁ | Vehicle Weight |
| x₂ | Engine Displacement |
| x₃ | Horsepower |
| w₁ | Weight coefficient |
| w₂ | Displacement coefficient |
| w₃ | Horsepower coefficient |
| b | Bias |
| y | Predicted Fuel Economy (MPG) |

---

## How Gradient Descent Works

During training, the model continuously improves by repeating the following steps.

1. Predict the fuel economy.
2. Calculate the prediction error.
3. Compute the gradients.
4. Update the weights and bias.
5. Repeat until the loss converges.

The update rule is

```
new_parameter = old_parameter - learning_rate × gradient
```

This optimization process allows the model to gradually learn the relationship between the vehicle's characteristics and its fuel economy.

---

## Training Workflow

| Step | Description |
|------|-------------|
| Prediction | Computes the predicted MPG |
| Loss Calculation | Calculates Mean Squared Error (MSE) |
| Gradient Computation | Computes gradients for every weight |
| Parameter Update | Updates weights and bias |
| Repeat | Continues until training is complete |

---

## Visualizations

The project produces two graphs.

### Loss vs Epoch

Displays how the Mean Squared Error decreases during training. A continuously decreasing curve indicates that Gradient Descent is successfully minimizing the error.

### Actual vs Predicted Fuel Economy

Compares the predicted fuel economy with the actual values from the dataset. Better predictions appear closer to the actual values.

---

## Sample Output

```
========== TRAINED MODEL ==========

Weights:
[-4.12 0.03 -0.08]

Bias:
52.67

Predicted Fuel Economy:
27.84 MPG
```

*The output above is only an example. Actual values depend on the dataset, learning rate, and training epochs.*

---

## Concepts Covered

| Topic | Included |
|--------|----------|
| Multiple Linear Regression | ✅ |
| Gradient Descent | ✅ |
| Matrix Operations | ✅ |
| Dot Product (`np.dot`) | ✅ |
| Matrix Transpose (`X.T`) | ✅ |
| Mean Squared Error (MSE) | ✅ |
| Weight Updates | ✅ |
| Bias Updates | ✅ |
| Fuel Economy Prediction | ✅ |
| Data Visualization | ✅ |

---

## Future Improvements

Possible enhancements for this project include:

- CSV dataset support
- Feature normalization
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Train/Test split
- Mini-batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Model saving and loading
- Real-world Auto MPG dataset
- Transition to Logistic Regression

---

## Author

**Jemelrey D. Abastillas**

Learning Machine Learning from scratch using NumPy to understand the mathematical foundations of modern AI before using high-level machine learning libraries.
