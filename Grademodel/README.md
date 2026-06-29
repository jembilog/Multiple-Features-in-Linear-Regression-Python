# Multiple Linear Regression from Scratch using NumPy

A simple implementation of **Multiple Linear Regression** built completely from scratch using **NumPy** and **Gradient Descent**. This project predicts a student's exam score using three input features: **study hours**, **attendance**, and **sleep hours**. The goal of this project is to understand how multiple linear regression works internally without relying on machine learning libraries such as scikit-learn.

---

## Dataset

The model is trained using three input features and one output.

| Feature | Description |
|---------|-------------|
| Study Hours | Number of hours the student studied |
| Attendance | Student attendance percentage |
| Sleep Hours | Number of hours of sleep |
| Exam Score | Target value to predict |

Example dataset:

| Study Hours | Attendance | Sleep Hours | Exam Score |
|-------------|------------|-------------|------------|
| 2 | 80 | 6 | 70 |
| 5 | 95 | 8 | 92 |
| 4 | 90 | 7 | 85 |
| 3 | 85 | 6 | 78 |
| 6 | 98 | 8 | 96 |

---

## Mathematical Model

Unlike simple linear regression,

```
y = mx + b
```

multiple linear regression learns multiple weights, one for each feature.

```
y = w₁x₁ + w₂x₂ + w₃x₃ + b
```

Where:

| Variable | Meaning |
|----------|---------|
| x₁ | Study Hours |
| x₂ | Attendance |
| x₃ | Sleep Hours |
| w₁ | Weight for Study Hours |
| w₂ | Weight for Attendance |
| w₃ | Weight for Sleep Hours |
| b | Bias |
| y | Predicted Exam Score |

---

## How Gradient Descent Works

The model improves itself by repeating the following steps thousands of times.

1. Predict the exam scores.
2. Calculate the prediction error.
3. Compute the gradients.
4. Update the weights and bias.
5. Repeat until the loss decreases.

The parameter update rule is

```
new_parameter = old_parameter - learning_rate × gradient
```

This process gradually moves the model toward the optimal solution.

---

## Training Process

For every epoch, the model performs:

| Step | Description |
|------|-------------|
| Prediction | Computes the predicted exam scores |
| Loss | Calculates Mean Squared Error (MSE) |
| Gradient | Computes gradients for each weight |
| Update | Updates the weights and bias |
| Repeat | Continues until training is complete |

---

## Visualizations

The project generates two graphs using Matplotlib.

### Loss vs Epoch

Shows how the Mean Squared Error decreases during training. A downward curve indicates that the model is learning successfully.

### Actual vs Predicted Scores

Compares the actual exam scores with the predicted scores after training. The closer the predicted values are to the actual values, the better the model performs.

---

## Sample Output

```
Weights:
[1.72 0.64 2.15]

Bias:
4.38

Predicted Score:
88.43
```

*The values above are examples. Your results may vary depending on the dataset, learning rate, and number of training epochs.*

---

## Concepts Covered

This project demonstrates the following machine learning concepts.

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
| Prediction | ✅ |
| Data Visualization | ✅ |

---

## Future Improvements

Some possible extensions for this project include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- CSV dataset support
- Feature normalization
- Train/Test split
- Mini-batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Model saving and loading
- Transition to Logistic Regression

---

## Author

**Jemelrey D. Abastillas**

Learning Machine Learning from scratch using NumPy to understand the mathematics behind modern AI algorithms before using high-level machine learning frameworks.
