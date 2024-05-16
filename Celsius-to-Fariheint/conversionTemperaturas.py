import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

# Importando datos
datosTemperature = pd.read_csv("celsius_a_fahrenheit.csv")

# Visualizacion
sbn.scatterplot( data=datosTemperature, x=datosTemperature["Celsius"], y=datosTemperature["Fahrenheit"] )


# Cargando Set de datos
X_train = datosTemperature['Celsius']
Y_train = datosTemperature['Fahrenheit']

# Creación de modelo
model = tf.keras.Sequential()

# Capa de input (unit = 1; 1 neurona)
# Aqui configurlo la neurona, que solo requiero de una
model.add(tf.keras.layers.Dense(units = 1, input_shape=[1]) )

# Es para ver la red
# model.summary()

# Compilado
# Aquí compila la red neuronal, con un optimizador adam y verificando la perdidda del error
model.compile(optimizer=tf.keras.optimizers.Adam(1.0), loss='mean_squared_error')

# Entrenando el modelo
# Aqui muestra como va nuestro mdoelo, que tanto ha aprendido con el set de datos
epochsHistory = model.fit(X_train, Y_train, epochs=100)

# Evaluo la eficacia de mi bot (Graficando)
epochsHistory.history.keys()

# Grafico
plt.plot(epochsHistory.history['loss'])

# Coloco una descripcion, titulo, etc al grafico
plt.title('Progreso de pérdida durante el entrenamiento del modelo')
plt.xlabel('Epoch')
plt.ylabel('Trainning Loss')
plt.legend('Trainnning Loss')


# Pesos 
model.get_weights()

# Predicciones

#Input
TemperatureIn = 0

#Output
TemperatureOu = model.predict( [TemperatureIn])
print(TemperatureOu)

# Aqui hago la prueba de la red neuronal apra que se acerque al 32
TemperaturaOu = 9/5 * TemperatureIn + 32
print(TemperaturaOu)



