from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import numpy as np

# calcula o coeficiente de correlação
r = pearsonr(X, Y)
print(r)

# separa os dados de treino e de teste (70% e 30%, respectivamente)
x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.3)

# redimensiona os dados em arrays bidimensionais
x_train = np.reshape(x_train, (-1, 1))
y_train = np.reshape(y_train, (-1, 1))
x_test = np.reshape(x_test, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

# treina o modelo
line = LinearRegression()
line.fit(x_train, y_train)
line_pred = line.predict(x_test)

# calcula o coeficiente r2
# retorna a porcentagem do quanto o modelo explica corretamente a variação da variável
r_squared = r2_score(y_test, line_pred)
print(f'Coeficiente r2: {r_squared}')