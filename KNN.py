import pandas as pd
import math
from collections import Counter

# Definir la función de distancia euclidiana
def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

# Definir la función para encontrar la moda (usada en clasificación)
def mode(labels):
    return Counter(labels).most_common(1)[0][0]

# Implementar el algoritmo KNN
def knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []

    # Calcular la distancia entre la consulta y cada ejemplo en los datos
    for index, example in enumerate(data):
        distance = distance_fn(example[:-1], query)
        neighbor_distances_and_indices.append((distance, index))

    # Ordenar los vecinos por distancia
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)

    # Seleccionar los K vecinos más cercanos
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]

    # Obtener las etiquetas de los K vecinos más cercanos
    k_nearest_labels = [data[i][-1] for _, i in k_nearest_distances_and_indices]

    # Retornar la predicción basada en la función de decisión
    return k_nearest_distances_and_indices, choice_fn(k_nearest_labels)

# Cargar el dataset
file_path = r'C:\Users\LOQ\Documents\proyecto_modelos\diabetes.csv'
data = pd.read_csv(file_path)

# Preparar los datos
features = data.iloc[:, :-1].values  # Todas las columnas excepto 'Outcome'
labels = data['Outcome'].values      # La columna 'Outcome'
formatted_data = [list(features[i]) + [labels[i]] for i in range(len(labels))]

# Número de vecinos a considerar
k = 5

# Inicializar variables para calcular el porcentaje de aciertos
correct_predictions = 0
total_predictions = len(data)

# Iterar sobre todos los datos y calcular predicciones
for index, row in data.iterrows():
    query = row.iloc[:-1].values  # Usar las características de la fila (sin la columna 'Outcome')
    true_label = row['Outcome']   # La etiqueta real

    # Ejecutar KNN
    _, prediction = knn(formatted_data, query, k, euclidean_distance, mode)

    # Comparar la predicción con la etiqueta real
    if prediction == true_label:
        correct_predictions += 1

# Calcular el porcentaje de aciertos
accuracy = (correct_predictions / total_predictions) * 100

# Imprimir resultado
print(f"Porcentaje de aciertos: {accuracy:.2f}%")
