import numpy as np
from neural_network import NeuralNetwork

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

print("Entrenando la red neuronal...\n")
nn.fit(X, y, learning_rate=0.03, epochs=40000)

print("\n--- Resultados ---")
for i, e in enumerate(X):
    pred = nn.predict(e)
    print(f"Entrada: {e}, Esperado: {y[i]}, Obtenido: {np.round(pred)}")
 