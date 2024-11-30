import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import roc_curve, auc
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import warnings

# Suprimir advertencias futuras
warnings.simplefilter(action='ignore', category=FutureWarning)

# Cargar el dataset de diabetes
file_path = r'C:\Users\LOQ\Documents\proyecto_modelos\diabetes.csv'
data = pd.read_csv(file_path)

# Verificar los primeros registros del dataset
print(data.head())

# Codificación de etiquetas si es necesario (en este caso Outcome ya es numérico)
# Si el dataset tiene valores categóricos que necesitan codificación, puedes hacer algo similar a lo siguiente:
# le = LabelEncoder()
# data['column_name'] = le.fit_transform(data['column_name'])

# Preparar los datos
X = data.drop(columns=['Outcome'])  # Características
y = data['Outcome']  # Etiquetas (diabetes o no)

# Dividir el dataset en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=22)

# Crear el modelo de árbol de decisión
dt = DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_leaf=8, min_samples_split=8, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
dt.fit(X_train, y_train)

# Predecir con los datos de prueba
y_pred = dt.predict(X_test)

# Calcular la precisión del modelo
accur = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accur * 100:.2f}%")

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(conf_matrix)

# Precisión y Recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Precisión: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")

# Reporte de clasificación (incluye precisión, recall, F1-score)
report = classification_report(y_test, y_pred)
print("Reporte de Clasificación:")
print(report)

# Curva ROC y AUC
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred)
roc_auc_dt = auc(fpr_dt, tpr_dt)

# Graficar la curva ROC
plt.figure(1)
lw = 2
plt.plot(fpr_dt, tpr_dt, color='green', lw=lw, label='Árbol de Decisión (AUC = %0.2f)' % roc_auc_dt)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under Curve')
plt.legend(loc="lower right")
plt.show()

# Visualizar el árbol de decisión
dot_data = StringIO()
export_graphviz(dt, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
