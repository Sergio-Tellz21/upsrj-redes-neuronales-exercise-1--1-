import numpy as np
from perceptron import Perceptron
from perceptron.input_data import InputData

def forward_propagation_network(inputs: np.ndarray, perceptrons: int, layers: int) -> float:
    # Convertimos las entradas en una lista de floats (por si vienen como np.array)
    entrada_actual = inputs.tolist()

    for _ in range(layers):
        capa_salidas = []

        for _ in range(perceptrons):
            entradas_neurona = [InputData(x) for x in entrada_actual]
            sesgo = np.random.randn()
            p = Perceptron(inputs=entradas_neurona, b=sesgo)
            capa_salidas.append(p.a)

        entrada_actual = capa_salidas  # Lo que sale se convierte en la entrada para la siguiente capa

    # Retornar un solo valor (por ejemplo, promedio de la última capa)
    return float(np.mean(entrada_actual))
