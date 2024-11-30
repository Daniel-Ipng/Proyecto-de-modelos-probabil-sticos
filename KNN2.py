import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Cargar el dataset de diabetes
file_path = r'C:\Users\LOQ\Documents\proyecto_modelos\diabetes.csv'
data = pd.read_csv(file_path)

# Verificar los primeros registros del dataset para tener una idea de su estructura
print(data.head())

# Preparar los datos
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]  # Características
y = data['Outcome']  # Etiquetas

# Crear el modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Entrenar el modelo con todos los datos
knn.fit(X, y)

# Realizar predicciones sobre los mismos datos (todos los valores)
y_pred = knn.predict(X)

# Calcular la precisión del modelo (porcentaje de aciertos)
accuracy = accuracy_score(y, y_pred)

# Imprimir la precisión
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Graficar los datos
ax = plt.axes()
ax.scatter(data.loc[data['Outcome'] == 1, 'BMI'], 
           data.loc[data['Outcome'] == 1, 'Age'], 
           c="red", label="Diabetes")
ax.scatter(data.loc[data['Outcome'] == 0, 'BMI'], 
           data.loc[data['Outcome'] == 0, 'Age'], 
           c="blue", label="No Diabetes")
plt.xlabel("BMI")
plt.ylabel("Age")
ax.legend()
plt.show()
