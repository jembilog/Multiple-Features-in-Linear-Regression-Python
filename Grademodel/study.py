import numpy as np
import matplotlib.pyplot as plt
X = np.array([
    [2,80,6],
    [5,95,8],
    [4,90,7],
    [3,85,6],
    [6,98,8]
])#Study Hours, Attendance, SLeep hours

Y = np.array([70,92,85,78,96])#exam scores

w = np.zeros(3) # -> w = [0,0,0]
b = 0
n= len(X)

learning_rate = 0.0001
epochs = 1000

losses = []

for epoch in range(epochs):
    Y_pred = np.dot(X, w) + b

    mse = np.mean((Y - Y_pred) ** 2)
    losses.append(mse)

    dw = (-2/n) * np.dot(X.T, (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db

new_student = np.array([4,92,7])
prediction = np.dot(new_student , w) + b
print("Weights: ", w)
print("Bias: ", b)
print(f"Predicted Score: {prediction:.4f}")

#gusto ko iplot dalawa, oh bakit
plt.figure(figsize=(8,5))
plt.plot(losses)
plt.title("Gradient Descent")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()


plt.figure(figsize=(8,5))
plt.plot(Y, marker='o',label='Actual')
plt.plot(Y_pred,marker='x',label='predicted')
plt.title("Actual vs Predicted Scores")
plt.xlabel("Student")
plt.ylabel("Exam score")
plt.legend()
plt.grid(True)
plt.show()