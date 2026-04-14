import os
import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure

load_dotenv()

WEAVIATE_MODE = os.getenv("WEAVIATE_MODE", "local").lower()
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "DepartmentChunk")


def get_weaviate_client():
    if WEAVIATE_MODE == "cloud":
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

        if not weaviate_url or not weaviate_api_key:
            raise RuntimeError("WEAVIATE_URL and WEAVIATE_API_KEY must be set for cloud mode")

        return weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )

    http_host = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
    http_port = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
    grpc_host = os.getenv("WEAVIATE_GRPC_HOST", http_host)
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

    return weaviate.connect_to_local(
        host=http_host,
        port=http_port,
        grpc_port=grpc_port,
    )


def _collection_properties():
    return [
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

        Property(name="content_source", data_type=DataType.TEXT),
        Property(name="content_type", data_type=DataType.TEXT),

        # Addded for catalog chunking
        Property(name="catalog_page", data_type=DataType.INT),
        Property(name="catalog_page_end", data_type=DataType.INT),
        Property(name="catalog_year", data_type=DataType.TEXT),
        Property(name="program_family", data_type=DataType.TEXT_ARRAY),
        Property(name="degree_level", data_type=DataType.TEXT),
        Property(name="degree_type", data_type=DataType.TEXT),
        Property(name="concentration", data_type=DataType.TEXT),
        Property(name="degree_full_title", data_type=DataType.TEXT),
        Property(name="credits", data_type=DataType.TEXT),
        Property(name="dept_prefix", data_type=DataType.TEXT),
        Property(name="course_number_level", data_type=DataType.TEXT),
        Property(name="has_prerequisites", data_type=DataType.BOOL),
        Property(name="policy_topic", data_type=DataType.TEXT),
        Property(name="lab_name", data_type=DataType.TEXT),
        Property(name="referenced_courses", data_type=DataType.TEXT_ARRAY),
        Property(name="is_research_related", data_type=DataType.BOOL),
    ]


def ensure_collection(client):
    properties = _collection_properties()

    if not client.collections.exists(WEAVIATE_COLLECTION):
        client.collections.create(
            WEAVIATE_COLLECTION,
            vector_config=Configure.Vectors.self_provided(),
            properties=properties,
        )
        return

    collection = client.collections.use(WEAVIATE_COLLECTION)
    existing = {prop.name for prop in collection.config.get().properties}

    for prop in properties:
        if prop.name not in existing:
            collection.config.add_property(prop)


def get_collection(client):
    return client.collections.use(WEAVIATE_COLLECTION)
