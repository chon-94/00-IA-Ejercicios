#Redes Neuronales para Regresión
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers


modelo_1=Sequential([  layers.Dense(5,activation="relu",input_shape=[3]),
                       layers.Dense(3,activation="relu"),
                       layers.Dense(1,activation=None)
                    ])
modelo_1.summary()
 