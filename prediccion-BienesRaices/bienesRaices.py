import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

# Importando datos
dataBienesRaices = pd.read_csv("precios_hogares.csv")

# Visualizacion
sbn.scatterplot( data=dataBienesRaices, x=dataBienesRaices["sqft_living"], y=dataBienesRaices["price"])

