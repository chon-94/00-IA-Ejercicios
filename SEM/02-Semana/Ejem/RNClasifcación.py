#Redes Neuronales para Clasifcación
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers

modelo_2=Sequential([  layers.Dense(5,activation="relu",input_shape=[3]),
                       layers.Dense(3,activation="relu"),
                       layers.Dense(1,activation="sigmoid")
                    ])

modelo_2.summary()