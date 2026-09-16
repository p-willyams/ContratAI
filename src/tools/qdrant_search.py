import os

import dotenv
import qdrant_client
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import models

dotenv.load_dotenv()

QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")

DENSE_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "Qdrant/BM25"
COLBERT_MODEL = "jinaai/jina-colbert-v2"

CATEGORIAS_VALIDAS = {
    "objeto",
    "prazo",
    "pagamento",
    "rescisao",
    "multa",
    "confidencialidade",
    "propriedade_intelectual",
    "foro",
}

GLOSSARIO_CATEGORIA = {
    "objeto": "Definição do objeto do contrato e escopo do serviço prestado",
    "prazo": "Duração, prazo de vigência e renovação do contrato",
    "pagamento": "Forma e prazo de pagamento, juros e correção por atraso",
    "rescisao": "Rescisão, denúncia e encerramento do contrato",
    "multa": "Multa, penalidade e cláusula penal por descumprimento ou rescisão antecipada do contrato",
    "confidencialidade": "Confidencialidade, sigilo e proteção de dados pessoais",
    "propriedade_intelectual": "Propriedade intelectual e direitos autorais sobre o trabalho produzido",
    "foro": "Foro, mediação e resolução de disputas contratuais",
}

_client = qdrant_client.QdrantClient(api_key=QDRANT_API_KEY, url=QDRANT_CLUSTER_URL)
_dense_embedding = TextEmbedding(DENSE_MODEL)
_sparse_embedding = SparseTextEmbedding(SPARSE_MODEL)
_colbert_embedding = LateInteractionTextEmbedding(COLBERT_MODEL)


def _formatar_resultado(ponto) -> str:
    meta = ponto.payload.get("metadata", {})
    linhas = [
        f"[{meta.get('tipo', 'tipo não informado')} | categoria: {meta.get('categoria', 'não informada')}]",
        f"Referência: {meta.get('referencia', 'não informada')}",
    ]
    if meta.get("link_original"):
        linhas.append(f"Link: {meta['link_original']}")
    linhas.append(f"Trecho: {ponto.payload.get('text', '')}")
    return "\n".join(linhas)


def busca_juridica(pergunta: str, categoria: str | None = None, limite: int = 5) -> str:
    if categoria and categoria not in CATEGORIAS_VALIDAS:
        return (
            f"Categoria '{categoria}' inválida. Categorias aceitas: "
            f"{', '.join(sorted(CATEGORIAS_VALIDAS))}. Tente novamente sem "
            f"o filtro de categoria ou com um valor válido."
        )

    texto_busca = pergunta
    if categoria:
        texto_busca = f"{GLOSSARIO_CATEGORIA[categoria]}. {pergunta}"

    dense_query = list(_dense_embedding.query_embed([texto_busca]))[0].tolist()
    sparse_query = list(_sparse_embedding.query_embed([texto_busca]))[0].as_object()
    colbert_query = list(_colbert_embedding.query_embed([texto_busca]))[0].tolist()

    filtro = None
    if categoria:
        filtro = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.categoria", match=models.MatchValue(value=categoria)
                )
            ]
        )

    resultados = _client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                prefetch=[
                    models.Prefetch(
                        query=dense_query, using="dense", limit=20, filter=filtro
                    ),
                    models.Prefetch(
                        query=sparse_query, using="sparse", limit=20, filter=filtro
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=20,
                filter=filtro,
            )
        ],
        query=colbert_query,
        using="colbert",
        limit=limite,
        query_filter=filtro,
    )

    if not resultados.points:
        return (
            "Nenhum resultado encontrado na base jurídica para esta pergunta. "
            "Isso pode significar que o assunto está fora do escopo desta base "
            "(contratos de prestação de serviço PJ - Código Civil, LGPD e "
            "Lei 9.610/98). Não invente uma resposta legal sem fonte."
        )

    return "\n\n---\n\n".join(_formatar_resultado(p) for p in resultados.points)


if __name__ == "__main__":
    print(busca_juridica("multa por rescisão antecipada", categoria="multa"))
