import tensorflow as tf
import numpy as np

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print("Comenzar entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo Entrenado!")

import matplotlib.pyplot as plt
#grafica
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de Perdida")
plt.plot(historial.history["loss"])
plt.show()
#prediccion
print("Hagamos una prueba de prediccion")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " fahrenheit")
#ver los valores asignados
print("Variables internas del modelo")
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())