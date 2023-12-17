import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#Datos de aprendizaje
x = [1, 1, 1.5, 2, 2, 2, 3, 3, 3, 3.5, 3.6,4, 5.1, 5.6, 6.1, 7]
y = [12, 7, 9, 5, 9, 11, 2, 6, 10, 8, 3, 2, 3, 4, 1, 2]
classes = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]

data = list(zip(x, y))
print("--------")
print("Datos de aprendizaje")
print(data)
print("--------")

#Definición de K
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(data, classes)

#Nuevo putno/s
newA_x = 2.5
newA_y = 7
newB_x = 5.5
newB_y = 4.5
new_point = [(newA_x, newA_y), (newB_x, newB_y)]

#Fase de clasificación
print("--------")
print("Array con nuevos datos: ", new_point)
prediction = knn.predict(new_point)

#Resultado
print("--------")
print("La clasificación de los siguientes puntos es:")
print(new_point[0], " = ", prediction[0])
print(new_point[1], " = ", prediction[1])
