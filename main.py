import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import numpy as np


dados = pd.read_csv('sheet3.csv')
X = dados["Po"]
Y = dados["Agua"]

# gráfico de dispersão
plt.scatter(X, Y)
plt.title("Água vs Suco em Pó")
plt.xlabel("Suco em Pó (Mg)")
plt.ylabel("Água (Ml)")
plt.show()

# calcula o coeficiente de correlação
r = pearsonr(X, Y)
print(r)

# separa os dados de treino e de teste (70% e 30%, respectivamente)
x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.3)

# redimensiona os dados em arrays bidimensionais
x_train = x_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# treina o modelo
line = LinearRegression()
line.fit(x_train, y_train)
line_pred = line.predict(x_test)

# calcula o coeficiente r2

