from functools import lru_cache

import numpy as np


# Función para obtener el embedding de una frase utilizando la API de OpenAI
@lru_cache(3000)
def get_embedding(text, model="text-embedding-3-small"):
    import openai

    client = openai.OpenAI()
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def normalize(X):
    # Calcular la norma L2 de cada vector de características
    norms = np.linalg.norm(X, axis=1, keepdims=True)

    # Dividir cada vector de características por su norma L2
    X_normalized = X / norms

    return X_normalized


# Implementación del PCA desde cero
def pca(X, n_components=2):
    # Centrar los datos
    X_centered = X - np.mean(X, axis=0)

    # Calcular la matriz de covarianza
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Calcular los autovalores y autovectores de la matriz de covarianza
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Ordenar los autovectores por los autovalores descendentes
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Seleccionar los primeros n_components autovectores
    principal_components = eigenvectors[:, :n_components]

    # Proyectar los datos originales en el espacio de componentes principales
    X_pca = np.dot(X_centered, principal_components)

    return X_pca


# Frases de ejemplo
cute = [
    "El perro corre en el parque",
    "El gato duerme en el sofá",
    "El pájaro vuela en el cielo",
    "El pez nada en el agua",
    "La mariposa revolotea en el jardín",
]
python_code = [
    "def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)",
    "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
    "def binary_search(arr, x):\n    low = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == x:\n            return mid\n        elif arr[mid] < x:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
]
maths = [
    "The Pythagorean theorem states that in a right-angled triangle, the square of the length of the hypotenuse is equal to the sum of the squares of the lengths of the other two sides.",
    "La fórmula para calcular el área de un círculo es: $A = \\pi r^2$, donde $A$ es el área y $r$ es el radio.",
    "The quadratic formula is used to solve quadratic equations. It is given by: $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$, where $a$, $b$, and $c$ are the coefficients of the quadratic equation $ax^2 + bx + c = 0$.",
    "El teorema fundamental del cálculo establece una relación entre la derivada y la integral de una función. Se expresa como: $\\int_a^b f(x) dx = F(b) - F(a)$, donde $F(x)$ es una antiderivada de $f(x)$.",
    "La serie de Fibonacci es una sucesión de números en la que cada número es la suma de los dos anteriores. Los primeros términos son: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...",
    "The Euler's identity is a remarkable mathematical formula that connects the fundamental constants $e$, $i$, and $\\pi$. It is expressed as: $e^{i\\pi} + 1 = 0$.",
]
emojis = [
    "❤️",
    "✅",
    "✨",
    "🔥",
    "😊",
    "😂",
]

tweets = [
    "Just had an amazing workout at the gym! 💪 #fitness #motivation",
    "Can't wait for the weekend! Going on a hiking adventure with friends. 🌿⛰️ #outdoors #nature",
    "Excited to announce that I'll be speaking at the upcoming tech conference! 🎙️ #technology #conference",
    "Trying out a new recipe tonight. Fingers crossed it turns out delicious! 🍳👨‍🍳 #cooking #foodie",
    "Watching the sunset at the beach. The colors are breathtaking! 🌅 #beach #sunset #beauty",
    "Just finished reading an incredible book. Highly recommend it to everyone! 📚 #reading #bookworm",
]


__all__ = [
    "get_embedding",
    "cute",
    "python_code",
    "maths",
    "emojis",
    "tweets",
]
