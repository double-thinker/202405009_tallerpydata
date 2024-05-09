from matplotlib import pyplot as plt

from tools.embeddings import get_embedding, normalize, pca


def viz_embeddings(phrases, max_chars=15):
    # Obtener los embeddings de las frases
    embeddings = [get_embedding(phrase) for phrase in phrases]

    # Normalizar los embeddings
    embeddings = normalize(embeddings)

    # Aplicar PCA para reducir la dimensionalidad a 2
    embeddings_pca = pca(embeddings)

    # Graficar los embeddings en un plano 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])

    # Agregar etiquetas a los puntos
    for i, phrase in enumerate(phrases):
        plt.annotate(
            phrase[:max_chars],
            (embeddings_pca[i, 0], embeddings_pca[i, 1]),
            fontsize=10,
        )

    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title("Representación de Embeddings con PCA")
    plt.grid(True)
    plt.show()


__all__ = ["viz_embeddings"]
