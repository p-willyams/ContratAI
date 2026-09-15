import os
import json
import uuid
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
import dotenv
from tqdm import tqdm

dotenv.load_dotenv()
import qdrant_client
from qdrant_client import models

QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")

DENSE_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "Qdrant/BM25"
COLBERT_MODEL = "jinaai/jina-colbert-v2"

client = qdrant_client.QdrantClient(api_key=QDRANT_API_KEY, url=QDRANT_CLUSTER_URL)

collections = [c.name for c in client.get_collections().collections]

if QDRANT_COLLECTION_NAME not in collections:
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "colbert": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="metadata.categoria",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="metadata.tipo",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def id_para_uuid(id_original: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_original))


def ingest_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        legislacao_records = json.load(f)

    dense_embedding = TextEmbedding(DENSE_MODEL)
    sparse_embedding = SparseTextEmbedding(SPARSE_MODEL)
    colbert_embedding = LateInteractionTextEmbedding(COLBERT_MODEL)

    points = []

    for item in tqdm(legislacao_records, desc="Ingerindo legislação"):
        texto = item.get("texto")
        if not texto or not texto.strip():
            continue

        dense_vector = list(dense_embedding.passage_embed([texto]))[0].tolist()
        sparse_vector = list(sparse_embedding.passage_embed([texto]))[0].as_object()
        colbert_vector = list(colbert_embedding.passage_embed([texto]))[0].tolist()

        metadata = {
            "id": item.get("id"),
            "tipo": item.get("tipo"),
            "fonte_lei": item.get("fonte_lei"),
            "referencia": item.get("referencia"),
            "categoria": item.get("categoria"),
            "link_original": item.get("link_original"),
        }

        point = models.PointStruct(
            id=id_para_uuid(item.get("id")),
            vector={
                "dense": dense_vector,
                "sparse": sparse_vector,
                "colbert": colbert_vector,
            },
            payload={"text": texto, "metadata": metadata},
        )

        points.append(point)

        if len(points) >= 32:
            client.upload_points(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points,
                batch_size=5,
                parallel=1,
                max_retries=3,
            )
            points = []

    if points:
        client.upload_points(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points,
            batch_size=5,
            parallel=1,
            max_retries=3,
        )

    print("Upload completed!")


data_dir = os.path.join("../data")

ingest_json(os.path.join(data_dir, "leg_contratai.json"))
ingest_json(os.path.join(data_dir, "clausulas.json"))
