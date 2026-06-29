import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [2.3,140,90],
    [2.8,160,110],
    [3.1,180,120],
    [3.5,220,150],
    [3.8,250,170],
    [4.0,280,190],
    [4.3,300,210],
    [4.6,330,230]
])

Y = np.array([
    36,
    33,
    31,
    28,
    25,
    23,
    21,
    19
])

w = np.zeros(3)
b = 0

learning_rate = 0.000001
epochs = 5000

losses = []

n = len(X)

for epoch in range(epochs):
    Y_pred = np.dot(X, w) + b

    mse = np.mean((Y - Y_pred) ** 2)
    losses.append(mse)

    dw = (-2/n) * np.dot(X.T ,(Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)

    w-= learning_rate * dw
    b-= learning_rate * db

Y_pred = np.dot(X,w) + b

new_car = np.array([3.4,210,140])
prediction = np.dot(new_car , w) + b

print("\n========= TRAINED MODEL =========")
print("Weights: ",w)
print("Bias: ",b)
print(f"Predicted fuel economy: {prediction:.2f} MPG")
print("\nctual vs Predicted")

for actual, pred in zip(Y, Y_pred):
    print(f"Actual: {actual:.2f} | Predicted: {pred:.2f} MPG")

plt.figure(figsize=(8,5))
plt.plot(losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.grid(True)
plt.show()


plt.figure(figsize=(8,5))
plt.plot(Y, marker='o',label="Actual")
plt.plot(Y_pred, marker='x',label='Predicted')#PALITAN MO ITO kung gusto mo yungnew input feature na na sa line 51
plt.title("Actual vs Predicted Fuel Economy")
plt.xlabel("Car")
plt.ylabel("Miles per Gallon")
plt.legend()
plt.grid(True)
plt.show()

