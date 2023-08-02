import numpy as np                    #matrices y funciones matematicas
import matplotlib.pyplot as plt       #gráficos y ploteos
import tensorflow as tf               #redes neuronales
#Modelo de una neurona artificial
from tensorflow import keras
from tensorflow.keras import Sequential,layers

x=np.array([2,3,4,5,6,7,8,9])
y=np.array([2.1,3.2,4.1,4.9,6.2,7.1,7.8,9.1] )

modelo=Sequential([layers.Dense(units=1,input_shape=[1],activation=None)])
modelo.compile(optimizer="sgd",loss="mean_squared_error")

hist=modelo.fit(x,y,epochs=100)

loss_mse=hist.history["loss"]
epochs=range(1,len(loss_mse)+1)

plt.plot(epochs,loss_mse)
plt.grid()
plt.show()

#Predicción
x_new=np.array([10.0])
y_pred=modelo.predict(x_new)
print("La predicción es: ",y_pred)

#Pesos sinápticos
print(modelo.get_weights())