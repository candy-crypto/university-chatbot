import os
import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure

load_dotenv()

WEAVIATE_MODE = os.getenv("WEAVIATE_MODE", "local").lower()
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "DepartmentChunk")


def get_weaviate_client():
    """
    Create and return a Weaviate client.
    Supports:
      - local Weaviate
      - Weaviate Cloud
    """
    if WEAVIATE_MODE == "cloud":
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

        if not weaviate_url or not weaviate_api_key:
            raise RuntimeError("WEAVIATE_URL and WEAVIATE_API_KEY must be set for cloud mode")

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        return client

    # Default: local
    http_host = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
    http_port = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
    grpc_host = os.getenv("WEAVIATE_GRPC_HOST", http_host)
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

    client = weaviate.connect_to_local(
        host=http_host,
        port=http_port,
        grpc_port=grpc_port,
    )
    return client


def ensure_collection(client):
    """
    Create the collection if it does not exist.
    This uses self-provided vectors, because embeddings come from OpenAI.
    """
    if client.collections.exists(WEAVIATE_COLLECTION):
        return

    client.collections.create(
        WEAVIATE_COLLECTION,
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name="department_id", data_type=DataType.TEXT),
            Property(name="document_id", data_type=DataType.TEXT),
            Property(name="url", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="section", data_type=DataType.TEXT),
            Property(name="timestamp", data_type=DataType.DATE),
            Property(name="tags", data_type=DataType.TEXT_ARRAY),
            Property(name="course_number", data_type=DataType.TEXT),
            Property(name="course_title", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="crawl_version", data_type=DataType.TEXT),
        ],
    )


def get_collection(client):
    return client.collections.use(WEAVIATE_COLLECTION)
