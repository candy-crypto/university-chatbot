# export_collection.py
"""Export the entire DepartmentChunk Weaviate collection — properties and vectors.

Candy runs this script on her machine (where Weaviate is running) to produce
a portable snapshot that Barbara can import on her own machine.
No re-embedding required: vectors are included in the export.

Usage (from the backend/ directory):
    python eval/export_collection.py
    python eval/export_collection.py --output department_chunk_export.json

Output:
    A JSON file containing a list of objects, each with:
      - "uuid":       the Weaviate object UUID (preserved on import)
      - "properties": all chunk properties (text, chunk_id, source, etc.)
      - "vector":     the full embedding vector (list of floats)

File size: roughly 15-25 MB depending on collection size.
Transfer via Google Drive, USB, or any file-sharing method.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weaviate_client import get_weaviate_client, ensure_collection, get_collection


DEFAULT_OUTPUT = Path(__file__).parent / "department_chunk_export.json"


def main():
    parser = argparse.ArgumentParser(
        description="Export DepartmentChunk collection with vectors to JSON"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT),
        help="Path to write the JSON export (default: eval/department_chunk_export.json)"
    )
    args = parser.parse_args()

    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = get_collection(client)

        print("Counting objects...")
        count_result = collection.aggregate.over_all(total_count=True)
        total = count_result.total_count
        print(f"Exporting {total} objects (this may take a moment)...\n")

        objects = []
        for i, item in enumerate(collection.iterator(include_vector=True), start=1):
            objects.append({
                "uuid":       str(item.uuid),
                "properties": item.properties,
                "vector":     item.vector.get("default") if isinstance(item.vector, dict) else item.vector,
            })
            if i % 100 == 0:
                print(f"  {i}/{total} exported...")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(objects, f, ensure_ascii=False)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nExported {len(objects)} objects to {output_path} ({size_mb:.1f} MB)")
        print("Transfer this file to Barbara's machine and run import_collection.py.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
