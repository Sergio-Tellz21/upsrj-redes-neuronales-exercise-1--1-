# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Redes Neuronales
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC03
# Archivo: perceptron.py
# Descripción: definición de una clase que abstrae un perceptrón
# ============================================================
import numpy as np
from perceptron.input_data import InputData

class Perceptron:
    def __init__(self, inputs: list[InputData], b: float):
        self.inputs = inputs
        self.b = b
        self.z = self.forward()
        self.a = self.activation()
        
    def forward(self):
        z = sum(input.x * input.w for input in self.inputs)
        return z + self.b
    
    def activation(self):
        return 1.0 / (1.0 + np.exp(-self.z))
