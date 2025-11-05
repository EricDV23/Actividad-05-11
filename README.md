# Actividad-05-11
#Arquitectura de Computadoras
#Velázquez Eric - Velázquez Bruno

Implementar la aplicaciòn en Arduino Uno en Wokwi del proyecto:
https://wokwi.com/projects/446823075914214401

1-Resumen de las arquitecturas observadas en el proyecto
En este proyecto se usan dos arquitecturas de red neuronal, una en Python y otra replicada en Arduino (ambas equivalentes, pero adaptadas a cada entorno):

Arquitectura en Python
Tipo: Red Neuronal Feedforward (propagación hacia adelante).

Estructura:

Capa de entrada: 2 neuronas (sensor de distancia y posición del obstáculo).

Capa oculta: 3 neuronas.

Capa de salida: 4 neuronas (una por motor del coche).

Función de activación: tanh (tangente hiperbólica).

Entrenamiento:

Se usó descenso del gradiente con tasa de aprendizaje de 0.03.

40.000 épocas de entrenamiento.

Objetivo: Aprender qué motores activar según las lecturas de los sensores para evitar obstáculos.

Arquitectura en Arduino
Tipo: Red Neuronal Feedforward preentrenada (solo inferencia).

Estructura:

Capa de entrada: 3 neuronas (2 entradas + 1 bias).

Capa oculta: 4 neuronas (incluye bias).

Capa de salida: 4 neuronas (una por motor).

Activación: tanh.

Proceso:

Los pesos entrenados en Python se copian al código de Arduino.

Arduino solo realiza la propagación hacia adelante para calcular salidas en tiempo real.

Interacción con hardware:

Las salidas controlan directamente los motores mediante un controlador L298N.

Entradas provienen del sensor ultrasónico y la posición del servo.

2-Enfoques de resolución de problemas aplicados
El proyecto aplica dos enfoques de resolución de problemas clave:

a) Aprendizaje supervisado
Se entrena una red neuronal con un conjunto de ejemplos (entradas y salidas esperadas).

El modelo “aprende” la relación entre sensores (entradas) y movimientos correctos (salidas).

Las salidas son binarias (encender/apagar motores), equivalentes a decisiones de acción.

b) Simulación – Transferencia de aprendizaje a hardware
El aprendizaje se hace en Python (ambiente de simulación).

Los pesos obtenidos se transfieren al microcontrolador Arduino.

En Arduino, se ejecuta únicamente la inferencia en tiempo real (sin reentrenar).

En resumen:
Primero se simula y entrena en software, luego se implementa en hardware real para controlar un vehículo autónomo físico.

3-Ejecutar el colab para entrenar la red neuronal

[5]
2 s
import numpy as np

# --------------------------------------------
# Clase NeuralNetwork
# --------------------------------------------
class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        self.activation = activation
        self.weights = []
        self.biases = []
        self.num_layers = len(layers)

        # Inicialización de pesos y biases
        for i in range(self.num_layers - 1):
            weight = np.random.randn(layers[i], layers[i+1]) * 0.1
            bias = np.zeros((1, layers[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def _activate(self, x):
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise Exception("Función de activación no soportada")

    def _activate_derivative(self, x):
        if self.activation == 'tanh':
            return 1.0 - np.tanh(x)**2
        elif self.activation == 'sigmoid':
            fx = 1 / (1 + np.exp(-x))
            return fx * (1 - fx)
        else:
            raise Exception("Función de activación no soportada")

    def fit(self, X, y, learning_rate=0.03, epochs=40001):
        for epoch in range(epochs):
            # Forward propagation
            a = [X]
            z = []
            for w, b in zip(self.weights, self.biases):
                z.append(np.dot(a[-1], w) + b)
                a.append(self._activate(z[-1]))

            # Backpropagation
            error = y - a[-1]
            deltas = [error * self._activate_derivative(z[-1])]

            for i in range(self.num_layers - 2, 0, -1):
                delta = np.dot(deltas[-1], self.weights[i].T) * self._activate_derivative(z[i-1])
                deltas.append(delta)
            deltas.reverse()

            # Actualización de pesos y biases
            for i in range(len(self.weights)):
                self.weights[i] += learning_rate * np.dot(a[i].T, deltas[i])
                self.biases[i] += learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

            # Mostrar error cada cierto tiempo
            if epoch % 10000 == 0:
                loss = np.mean(np.square(error))
                print(f"Época {epoch} - Error medio: {loss:.6f}")

    def predict(self, X):
        a = X
        for w, b in zip(self.weights, self.biases):
            a = self._activate(np.dot(a, w) + b)
        return a


# --------------------------------------------
# Entrenamiento del modelo
# --------------------------------------------

# Crear red neuronal: 2 neuronas de entrada, 3 ocultas y 4 de salida
nn = NeuralNetwork([2, 3, 4], activation='tanh')

# Entradas (sensores del coche)
X = np.array([
    [-1, 0],   # sin obstáculos
    [-1, 1],   # sin obstáculos
    [-1, -1],  # sin obstáculos
    [0, -1],   # obstáculo derecha
    [0, 1],    # obstáculo izquierda
    [0, 0],    # obstáculo centro
    [1, 1],    # muy cerca derecha
    [1, -1],   # muy cerca izquierda
    [1, 0]     # muy cerca centro
])

# Salidas esperadas (motores)
# Cada vector [m1, m2, m3, m4] representa el estado de los motores
y = np.array([
    [1, 0, 0, 1], # avanzar
    [1, 0, 0, 1], # avanzar
    [1, 0, 0, 1], # avanzar
    [0, 1, 0, 1], # giro derecha
    [1, 0, 1, 0], # giro izquierda
    [1, 0, 0, 1], # avanzar
    [0, 1, 1, 0], # retroceder
    [0, 1, 1, 0], # retroceder
    [0, 1, 1, 0]  # retroceder
])

# Entrenar la red
print("Entrenando la red neuronal...\n")
nn.fit(X, y, learning_rate=0.03, epochs=40001)

# --------------------------------------------
# Evaluación de resultados
# --------------------------------------------

def valNN(x):
    """Redondea y convierte a 0 o 1 la salida"""
    return int(abs(round(x)))

print("\n--- Resultados ---")
for i, entrada in enumerate(X):
    prediccion = nn.predict(entrada)
    salida_red = [valNN(p) for p in prediccion[0]]
    print(f"Entrada: {entrada} | Esperado: {y[i]} | Obtenido: {salida_red}")

Entrenando la red neuronal...

Época 0 - Error medio: 0.498371
Época 10000 - Error medio: 0.000484
Época 20000 - Error medio: 0.000227
Época 30000 - Error medio: 0.000146
Época 40000 - Error medio: 0.000107

--- Resultados ---
Entrada: [-1  0] | Esperado: [1 0 0 1] | Obtenido: [1, 0, 0, 1]
Entrada: [-1  1] | Esperado: [1 0 0 1] | Obtenido: [1, 0, 0, 1]
Entrada: [-1 -1] | Esperado: [1 0 0 1] | Obtenido: [1, 0, 0, 1]
Entrada: [ 0 -1] | Esperado: [0 1 0 1] | Obtenido: [0, 1, 0, 1]
Entrada: [0 1] | Esperado: [1 0 1 0] | Obtenido: [1, 0, 1, 0]
Entrada: [0 0] | Esperado: [1 0 0 1] | Obtenido: [1, 0, 0, 1]
Entrada: [1 1] | Esperado: [0 1 1 0] | Obtenido: [0, 1, 1, 0]
Entrada: [ 1 -1] | Esperado: [0 1 1 0] | Obtenido: [0, 1, 1, 0]
Entrada: [1 0] | Esperado: [0 1 1 0] | Obtenido: [0, 1, 1, 0]
4-Simular 2 entradas nuevas y 1 salida más. Modificar y ajustar la red neuronal
Vamos a extender la red para que acepte dos entradas adicionales y una salida extra.

Por ejemplo:

Entradas nuevas:

Sensor de luz (para frenar en zonas oscuras).

Sensor de temperatura (para no sobrecalentar motores).

Nueva salida:

Motor ventilador (1 = encendido, 0 = apagado).

Nueva arquitectura propuesta
Entradas: 4 neuronas (Distancia, Posición, Luz, Temperatura).

Ocultas: 4 neuronas.

Salidas: 5 neuronas (4 motores + ventilador).

Activación: tanh.

Código Python ajustado:

[4]
import numpy as np
# Nueva red: 4 entradas, 3 ocultas, 5 salidas
nn = NeuralNetwork([4, 3, 5], activation='tanh')

# Nuevas entradas simuladas (distancia, posición, luz, temperatura)
X = np.array([
    [-1, 0,  1,  0],   # sin obstáculo, buena luz, temp normal
    [0, 1,  0,  0],    # obstáculo izq, sombra
    [1, 0, -1,  1],    # obstáculo centro, alta temp
    [0, -1, 1, -1]     # obstáculo derecha, luz alta, baja temp
])

# Salidas (motores y ventilador)
# [M1, M2, M3, M4, Ventilador]
y = np.array([
    [1, 0, 0, 1, 0],   # avanzar sin ventilar
    [1, 0, 1, 0, 0],   # giro izquierda
    [0, 1, 1, 0, 1],   # retroceder + ventilador on
    [0, 1, 0, 1, 0]    # giro derecha
])

# Entrenar la red
print("Entrenando la red neuronal...\n")
nn.fit(X, y, learning_rate=0.03, epochs=40000)

# Ver resultados
print("\n--- Resultados ---")
for i, e in enumerate(X):
    pred = nn.predict(e)
    print(f"Entrada: {e}, Esperado: {y[i]}, Obtenido: {np.round(pred)}")

Entrenando la red neuronal...

Época 0 - Error medio: 0.450719
Época 10000 - Error medio: 0.000054
Época 20000 - Error medio: 0.000026
Época 30000 - Error medio: 0.000017

--- Resultados ---
Entrada: [-1  0  1  0], Esperado: [1 0 0 1 0], Obtenido: [[ 1.  0.  0.  1. -0.]]
Entrada: [0 1 0 0], Esperado: [1 0 1 0 0], Obtenido: [[ 1. -0.  1.  0.  0.]]
Entrada: [ 1  0 -1  1], Esperado: [0 1 1 0 1], Obtenido: [[ 0.  1.  1. -0.  1.]]
Entrada: [ 0 -1  1 -1], Esperado: [0 1 0 1 0], Obtenido: [[ 0.  1. -0.  1.  0.]]
La red ahora tiene:

4 entradas:

Sensor de distancia

Posición del obstáculo

Sensor de luz

Sensor de temperatura

5 salidas:

Motor 1

Motor 2

Motor 3

Motor 4

Ventilador (nuevo actuador)

Tabla de verdad:
Sensor Distancia	Posición Obstáculo	Luz	Temperatura	Motor 1	Motor 2	Motor 3	Motor 4	Ventilador	Acción esperada
-1	0	1	0	1	0	0	1	0	Avanzar
-1	1	1	0	1	0	1	0	0	Giro izquierda
-1	-1	1	0	0	1	0	1	0	Giro derecha
0	0	1	0	1	0	0	1	0	Avanzar lento
0	1	0	0	1	0	1	0	0	Giro izquierda (luz baja)
0	-1	0	0	0	1	0	1	0	Giro derecha (luz baja)
1	0	1	0	0	1	1	0	0	Retroceder
1	1	1	1	0	1	1	0	1	Retroceder con ventilador
1	-1	1	1	0	1	1	0	1	Retroceder con ventilador
0	0	1	1	1	0	0	1	1	Avanzar con ventilador
-1	0	1	1	1	0	0	1	1	Avanzar con ventilador (temperatura alta)
Interpretación
Cuando la temperatura es alta (1), el ventilador se enciende (1).

Cuando la distancia es grande (-1), el coche avanza.

Cuando la distancia es corta (1), retrocede o gira.

La posición del obstáculo define hacia qué lado gira.

La luz baja puede hacer que el coche se mueva más lento o cambie el tipo de giro.
