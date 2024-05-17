import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

# Importando datos
dataBienesRaices = pd.read_csv("precios_hogares.csv")


if 'date' in dataBienesRaices.columns:
    dataBienesRaices['date'] = pd.to_datetime(dataBienesRaices['date'], format='%Y%m%dT%H%M%S')

# Verificar si hay columnas no numéricas
non_numeric_columns = dataBienesRaices.select_dtypes(exclude=[np.number]).columns

# Eliminar columnas no numéricas antes de calcular la correlación
dataBienesRaices_numeric = dataBienesRaices.select_dtypes(include=[np.number])

# Manejar valores NaN
dataBienesRaices_numeric = dataBienesRaices_numeric.dropna()

# Visualizacion
sbn.scatterplot( data=dataBienesRaices, x=dataBienesRaices["sqft_living"], y=dataBienesRaices["price"])

# Encontrar correlación
f, ax = plt.subplots( figsize = (20, 20) )
sbn.heatmap(dataBienesRaices_numeric.corr(), annot=True)


# Limpieza numeral de datos, los necesarios que ocupo para la predicción
# Y ocupar los datos con mayor correlación en mi heatmao
selectedData = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']

# Crear data frame x
x = dataBienesRaices_numeric[selectedData]
y = dataBienesRaices_numeric['price']

# Escalar datos de x
# Esto es para colocar datos de 0 a 1, para mejor visualización
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_Scaled = scaler.fit_transform(x)


# Normalizando OutPut (Y)
y = y.values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)


# Entrenamiento del modelo
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Scaled, y_scaled, test_size = 0.25)

# Definiendo modelo
model = tf.keras.models.Sequential()

# 3 capas con 100 neuronas cada una
model.add(tf.keras.layers.Dense(units=100, activation="relu", input_shape=(7, )))
model.add(tf.keras.layers.Dense(units=100, activation="relu"))
model.add(tf.keras.layers.Dense(units=100, activation="relu"))
# 1 sola neurona como capa de salida
model.add(tf.keras.layers.Dense(units=1, activation="linear"))

model.summary()


# Compilación de modelo
model.compile( optimizer="Adam", loss="mean_squared_error")

epochsHistory = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_split=0.2)



#Evaluando modelo
epochsHistory.history.keys()

# Grafico
plt.plot(epochsHistory.history['loss'])
plt.plot(epochsHistory.history['val_loss'])

# Coloco una descripcion, titulo, etc al grafico
plt.title('Progreso del modelo durante Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Trainning and Validation Loss')
plt.legend(['Trainnning Loss', 'Validation Loss'])


# Prediccion del modelo
# Definir una casa prueba
#Inputs
x_test_1 = np.array([[ 4, 3, 1960, 5000, 1, 2000, 3000]])

scaler_test_1 = MinMaxScaler()
scaler_test_1_scaled = scaler_test_1.fit_transform(x_test_1)

# Haciendo predicciones
y_predir1 = model.predict(scaler_test_1_scaled)


#Re ajustar el invesor de scaler
y_predictOk = scaler.inverse_transform(y_predir1)







