import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Importando datos
datosVentas = pd.read_csv("datos_de_ventas.csv")

# Visualizacion
sns.scatterplot( data=datosVentas, x=datosVentas['Temperature'], y=datosVentas["Revenue"])

# Creando set de datos
X_train = datosVentas['Temperature']
Y_train = datosVentas['Revenue']

# Creando modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 1, input_shape = [1]))

# Summary del modelo
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss = "mean_squared_error")

# Entrenamiento
epochsHistory = model.fit(X_train, Y_train, epochs=1000)
keys = epochsHistory.history.keys()

# Grafico de entrenamioento del modelo
plt.plot(epochsHistory.history['loss'])
plt.title('Progreso de pérdidda durante entrenamiento')
plt.xlabel('Epochs')
plt.ylabel('Trainning Loss')
plt.legend(['Trainning Loss'])


weights = model.get_weights()


# Prediccion

# Esto es el input que varía
Temperatura = 30
Revenue = model.predict([Temperatura])
print('La ganancia según la RRNN será de: ',Revenue)


# VAMOS A GRAFICAR LAS PREDICCIONES
plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, model.predict([X_train]), color="blue" )
plt.ylabel('Ganancia en dolares')
plt.xlabel('Temperatura en Grados Celsius')
plt.title('Ganancia generada vs Temperatura')