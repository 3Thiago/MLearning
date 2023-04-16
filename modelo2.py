import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Carrega o conjunto de dados
url = 'https://raw.githubusercontent.com/3Thiago/MLearning/main/voluminosos.csv'
data = pd.read_excel("voluminosos.xlsx")

# Separa as variáveis de entrada (X) e saída (y)
X = data.drop('Descricao_Item', axis=1)
y = data['Veiculo']

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria um modelo de árvore de decisão e ajusta aos dados de treinamento
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Faz previsões usando o conjunto de teste
y_pred = model.predict(X_test)

# Avalia a precisão do modelo usando a métrica de acurácia
accuracy = accuracy_score(y_test, y_pred)
print('Acurácia do modelo:', accuracy)
