# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Cargar el dataset de diabetes
file_path = r'C:\Users\LOQ\Documents\proyecto_modelos\diabetes.csv'
data = pd.read_csv(file_path)

# Verificar que los datos se cargaron correctamente
print(data.head())

# Preparar los datos
X = data.iloc[:, :-1].values  # Características (todas las columnas excepto 'Outcome')
y = data['Outcome'].values    # Etiquetas (columna 'Outcome')

# Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo con los datos de entrenamiento
clf.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Visualizar el árbol de decisión (opcional)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=data.columns[:-1], class_names=['No', 'Yes'], rounded=True)
plt.show()
