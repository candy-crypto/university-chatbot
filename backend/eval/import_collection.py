# import_collection.py
"""Import a DepartmentChunk export (from export_collection.py) into a local Weaviate instance.

Barbara runs this script after:
  1. Starting Weaviate locally with Docker (see instructions below)
  2. Receiving department_chunk_export.json from Candy

No OpenAI calls are made — vectors from the export are used directly.

── Docker setup (one time) ──────────────────────────────────────────────────

1. Install Docker Desktop: https://www.docker.com/products/docker-desktop/

2. Start Weaviate:
     docker run -d \
       --name weaviate \
       -p 8080:8080 \
       -p 50051:50051 \
       -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
       -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
       -v weaviate_data:/var/lib/weaviate \
       cr.weaviate.io/semitechnologies/weaviate:latest

3. Verify it's running:
     curl http://localhost:8080/v1/meta

   You should see a JSON response with Weaviate version info.

4. Set WEAVIATE_URL=http://localhost:8080 in your .env file
   (or it defaults to http://localhost:8080 if not set).

── Usage ────────────────────────────────────────────────────────────────────

From the backend/ directory:
    python eval/import_collection.py
    python eval/import_collection.py --input /path/to/department_chunk_export.json
    python eval/import_collection.py --clear   # wipe collection before importing

── Notes ────────────────────────────────────────────────────────────────────

- Run with --clear if you want a clean import (removes all existing objects first).
- UUIDs from the export are preserved, so re-importing is idempotent if
  the same export file is used.
- The script imports in batches of 100 to avoid memory issues.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weaviate.classes.data import DataObject
from weaviate_client import get_weaviate_client, ensure_collection, get_collection


DEFAULT_INPUT = Path(__file__).parent / "department_chunk_export.json"
BATCH_SIZE    = 100


def main():
    parser = argparse.ArgumentParser(
        description="Import DepartmentChunk export into local Weaviate"
    )
    parser.add_argument(
        "--input", "-i",
        default=str(DEFAULT_INPUT),
        help="Path to the JSON export file (default: eval/department_chunk_export.json)"
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Delete all existing objects from the collection before importing"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Export file not found: {input_path}")
        print("Run export_collection.py on Candy's machine first.")
        sys.exit(1)

    print(f"Loading export from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        objects = json.load(f)
    print(f"Loaded {len(objects)} objects.\n")

    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = get_collection(client)

        if args.clear:
            print("--clear specified: deleting all existing objects...")
            collection.data.delete_many(
                where=None  # Weaviate v4: delete_many with no filter deletes all
            )
            print("Collection cleared.\n")

        print(f"Importing in batches of {BATCH_SIZE}...")
        imported = 0
        errors   = 0

        for batch_start in range(0, len(objects), BATCH_SIZE):
            batch = objects[batch_start: batch_start + BATCH_SIZE]
            data_objects = []

            for obj in batch:
                vector = obj.get("vector")
                if vector is None:
                    print(f"  WARNING: no vector for UUID {obj.get('uuid')} — skipping")
                    errors += 1
                    continue

                data_objects.append(DataObject(
                    uuid=obj["uuid"],
                    properties=obj["properties"],
                    vector=vector,
                ))

            if data_objects:
                result = collection.data.insert_many(data_objects)
                batch_errors = len(result.errors) if result.errors else 0
                if batch_errors:
                    print(f"  Batch {batch_start // BATCH_SIZE + 1}: "
                          f"{len(data_objects) - batch_errors} ok, {batch_errors} errors")
                    errors += batch_errors
                imported += len(data_objects) - batch_errors

            done = min(batch_start + BATCH_SIZE, len(objects))
            print(f"  {done}/{len(objects)} processed...")

        print(f"\nImport complete: {imported} objects imported, {errors} errors.")
        if errors:
            print("Check output above for details on failed objects.")
        else:
            print("All objects imported successfully. Weaviate is ready for eval runs.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
