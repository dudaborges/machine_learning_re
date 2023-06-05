import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

dados = pd.read_csv('sheet3.csv')
X = dados["Po"]
Y = dados["Agua"]

reg = linear_model.LinearRegression()
reg.fit(X.values.reshape(-1, 1), Y)

A = reg.coef_
B = reg.intercept_
print(f"Coeficiente intercepto: {float(A)}")
print(f"Coeficiente angular: {float(B)}") 

def scatter_chart():
    plt.scatter(X, Y)
    plt.title("Água vs Suco em Pó")
    plt.xlabel("Suco em Pó (Mg)")
    plt.ylabel("Água (Ml)")
    x0 = dados["Po"][0]
    y0 = dados["Agua"][0]
    x1 = dados["Po"][3]
    y1 = dados["Agua"][3]
    plt.plot([x0, x1], [y0, y1], "r")
    plt.show()

scatter_chart()
def predict_value_x(Y):
    X = (Y - B) / A
    print(f"X é {float(X)}, se Y for {Y}")

predict_value_x(300)
def predict_value_y(X):
    Y = A * X + B
    print(f"Y é {float(Y)}, se X for {X}")