import json
import random
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient

from enterprise_rag.config import COLLECTION_NAME, QDRANT_URL
from eval.eval_config import EVAL_FOLDER, MAX_CHUNKS_PER_FILE, N_CHUNKS_TO_EXPORT, SEED

if __name__ == "__main__":
    # Riproducibility
    rng = random.Random(SEED)

    # Open connection
    client = QdrantClient(url=QDRANT_URL)

    # Count how many chunks
    collection_chunks = client.count(collection_name=COLLECTION_NAME, exact=True).count

    # Fetch all chunks
    content = client.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        with_vectors=False,
        # Default limit of the API is 10, so write the limit as the exact number of chunks to fetch all
        limit=collection_chunks,
    )[0]  # -> ([records], None)

    # First sample over chunks
    indices = rng.sample(range(collection_chunks), N_CHUNKS_TO_EXPORT)
    chunks = defaultdict(list)  # source: data
    for idx in indices:
        record = content[idx]
        chunks[Path(record.payload["metadata"]["source"]).stem].append(
            {
                "id": record.id,
                "text": record.payload["page_content"],
                "start_index": record.payload["metadata"]["start_index"],
            }
        )

    # Refinement to ensure the MAX_CHUNKS_PER_FILE is respected
    tot_extracted_chunks = 0

    # A source is a file
    for source in chunks:
        # Remove chunks from oversampled files
        while len(chunks[source]) > MAX_CHUNKS_PER_FILE:
            chunks[source].pop()
        tot_extracted_chunks += len(chunks[source])

    while tot_extracted_chunks < N_CHUNKS_TO_EXPORT:
        new_idx = int(rng.uniform(0, collection_chunks))
        new_record = content[new_idx]
        source = Path(new_record.payload["metadata"]["source"]).stem
        # Add a new chunk if either another file is selected or a chunk for a non saturated file
        if (
            source in chunks and len(chunks[source]) < MAX_CHUNKS_PER_FILE
        ) or source not in chunks:
            # Avoid duplicates
            skip_iter = False
            for el in chunks[source]:
                if el["id"] == new_record.id:
                    skip_iter = True
            if skip_iter:
                continue

            chunks[source].append(
                {
                    "id": new_record.id,
                    "text": new_record.payload["page_content"],
                    "start_index": new_record.payload["metadata"]["start_index"],
                }
            )
            tot_extracted_chunks += 1

    # Write the extraction on a jsonl file
    with Path.open(EVAL_FOLDER / "chunks_for_questions.jsonl", "w") as file:
        for source in chunks:
            for el in chunks[source]:
                record = {
                    "source": source,
                    "id": el["id"],
                    "text": el["text"],
                    "start_index": el["start_index"],
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
