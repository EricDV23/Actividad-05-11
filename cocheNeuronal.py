##Bruno Velazquez

import numpy as np

# Asumiendo que tenemos la clase NeuralNetwork definida
class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        self.layers = layers
        self.activation = activation
        # Inicializar pesos y biases...
        
    def fit(self, X, y, learning_rate=0.03, epochs=40001):
        # Implementación del entrenamiento...
        pass
        
    def predict(self, x):
        # Implementación de la predicción...
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Placeholder

# Red Coche para Evitar obstáculos - VERSIÓN MODIFICADA
# Nueva arquitectura: [4, 5, 5] (4 entradas, 5 neuronas ocultas, 5 salidas)
nn = NeuralNetwork([4, 5, 5], activation='tanh')

# DATASET MODIFICADO CON 4 ENTRADAS Y 5 SALIDAS
# Entradas: [sensor_izq, sensor_der, sensor_centro, velocidad_actual]
X = np.array([
    # Sin obstáculos, diferentes velocidades
    [-1, -1, -1, 0],   # sin obstáculos, parado
    [-1, -1, -1, 1],   # sin obstáculos, velocidad media
    [-1, -1, -1, 2],   # sin obstáculos, velocidad alta
    
    # Obstáculos detectados
    [0, -1, -1, 1],    # obstáculo izquierda, velocidad media
    [-1, 0, -1, 1],    # obstáculo derecha, velocidad media
    [-1, -1, 0, 1],    # obstáculo centro, velocidad media
    
    # Obstáculos muy cerca
    [1, -1, -1, 1],    # demasiado cerca izquierda
    [-1, 1, -1, 1],    # demasiado cerca derecha
    [-1, -1, 1, 1],    # demasiado cerca centro
    
    # Múltiples obstáculos
    [0, 0, -1, 1],     # obstáculos izquierda y derecha
    [1, 1, -1, 0],     # demasiado cerca ambos lados, parado
    [1, 1, 1, 1],      # demasiado cerca por todos lados
    
    # Situaciones con alta velocidad
    [-1, -1, -1, 2],   # sin obstáculos, alta velocidad
    [0, -1, -1, 2],    # obstáculo izquierda, alta velocidad
])

# SALIDAS MODIFICADAS CON 5 SALIDAS
# [motor_izq_adelante, motor_izq_atras, motor_der_adelante, motor_der_atras, luces_emergencia]
y = np.array([
    [1,0,0,1,0],  # avanzar normal
    [1,0,0,1,0],  # avanzar normal
    [1,0,0,1,0],  # avanzar normal
    
    [0,1,0,1,0],  # giro derecha (esquivar obstáculo izquierda)
    [1,0,1,0,0],  # giro izquierda (esquivar obstáculo derecha)
    [1,0,0,1,1],  # avanzar con precaución (luces emergencia)
    
    [0,1,0,1,1],  # retroceder y girar derecha con luces
    [1,0,1,0,1],  # retroceder y girar izquierda con luces  
    [0,1,1,0,1],  # retroceder con luces emergencia
    
    [0,1,1,0,1],  # retroceder (obstáculos ambos lados)
    [0,1,1,0,1],  # retroceder con luces
    [0,1,1,0,1],  # retroceder con luces (obstáculo total)
    
    [1,0,0,1,0],  # avanzar normal (alta velocidad)
    [0,1,0,1,1],  # giro derecha con luces (alta velocidad + obstáculo)
])

# Entrenar la red
nn.fit(X, y, learning_rate=0.03, epochs=40001)

def valNN(x):
    return (int)(abs(round(x)))

# TABLA DE VERDAD COMPLETA
print("=== TABLA DE VERDAD - SISTEMA MEJORADO ===")
print("Entradas: [sensor_izq, sensor_der, sensor_centro, velocidad]")
print("Salidas: [mot_izq_adel, mot_izq_atr, mot_der_adel, mot_der_atr, luces_emer]")
print("\n" + "="*80)

index = 0
for e in X:
    prediccion = nn.predict(e)
    prediccion_binaria = [valNN(prediccion[0]), valNN(prediccion[1]), 
                         valNN(prediccion[2]), valNN(prediccion[3]), 
                         valNN(prediccion[4])]
    
    # Descripción de la situación
    situacion = ""
    if e[0] == -1 and e[1] == -1 and e[2] == -1:
        situacion = "VÍA LIBRE"
    elif e[0] == 1 or e[1] == 1 or e[2] == 1:
        situacion = "PELIGRO INMINENTE"
    elif e[0] == 0 or e[1] == 0 or e[2] == 0:
        situacion = "OBSTÁCULO DETECTADO"
    
    velocidad = ["PARADO", "MEDIA", "ALTA"][int(e[3])]
    
    print(f"Situación: {situacion:20} | Velocidad: {velocidad:6}")
    print(f"Entrada:  {list(e)}")
    print(f"Esperado: {list(y[index])}")
    print(f"Obtenido: {prediccion_binaria}")
    
    # Interpretación de la acción
    accion = ""
    if prediccion_binaria == [1,0,0,1,0]:
        accion = "AVANZAR"
    elif prediccion_binaria == [0,1,0,1,0]:
        accion = "GIRO DERECHA"
    elif prediccion_binaria == [1,0,1,0,0]:
        accion = "GIRO IZQUIERDA" 
    elif prediccion_binaria == [0,1,1,0,0]:
        accion = "RETROCEDER"
    elif prediccion_binaria[4] == 1:
        accion = "EMERGENCIA - " + {
            (1,0,0,1,1): "AVANZAR CON PRECAUCIÓN",
            (0,1,0,1,1): "GIRO DERECHA EMERGENCIA",
            (1,0,1,0,1): "GIRO IZQUIERDA EMERGENCIA",
            (0,1,1,0,1): "RETROCEDER EMERGENCIA"
        }.get(tuple(prediccion_binaria), "ACCIÓN EMERGENCIA")
    
    print(f"Acción:   {accion}")
    print("-" * 80)
    index += 1