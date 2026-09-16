import os
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
import dotenv

dotenv.load_dotenv()
import qdrant_client
from qdrant_client import models

QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")

DENSE_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "Qdrant/BM25"
COLBERT_MODEL = "jinaai/jina-colbert-v2"

client_qdrant = qdrant_client.QdrantClient(
    api_key=QDRANT_API_KEY, url=QDRANT_CLUSTER_URL
)

dense_embedding = TextEmbedding(DENSE_MODEL)
sparse_embedding = SparseTextEmbedding(SPARSE_MODEL)
colbert_embedding = LateInteractionTextEmbedding(COLBERT_MODEL)


def build_context(results):
    blocks = []
    for i, r in enumerate(results.points):
        meta = r.payload.get("metadata", {})
        tipo = meta.get("tipo", "Tipo não informado")
        categoria = meta.get("categoria", "Categoria não informada")
        fonte_lei = meta.get("fonte_lei", "Fonte não informada")
        referencia = meta.get("referencia", "Referência não informada")
        link_original = meta.get("link_original", None)
        id_ = meta.get("id", "ID não informado")

        header = f"[Documento {i + 1}]"
        lines = [
            header,
            f"Tipo: {tipo}",
            f"Categoria: {categoria}",
            f"Fonte: {fonte_lei}",
            f"Referência: {referencia}",
            f"ID: {id_}",
        ]
        if link_original:
            lines.append(f"Link original: {link_original}")
        lines.append("Trecho:")
        lines.append(r.payload.get("text", ""))

        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


while True:
    query = input("Digite sua pergunta: ")

    dense_query = list(dense_embedding.passage_embed([query]))[0].tolist()
    sparse_query = list(sparse_embedding.passage_embed([query]))[0].as_object()
    colbert_query = list(colbert_embedding.passage_embed([query]))[0].tolist()

    results = client_qdrant.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch={
            "prefetch": [
                {"query": dense_query, "using": "dense", "limit": 20},
                {"query": sparse_query, "using": "sparse", "limit": 20},
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "limit": 20,
        },
        query=colbert_query,
        using="colbert",
        limit=10,
    )

    context = build_context(results)

    print("\n" + "=" * 40 + "\n")
    print("CONTEXTOS RECUPERADOS DAS QUERYS:\n")
    print(context)
    print("\n" + "=" * 40 + "\n")
