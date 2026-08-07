# TODO: provide a clean way to push data in the database
# QUESTO SOTTO è COME ANDREBBE FATTO MA è PRIMA DI USARE TUTTE LE VARIABILI DI CONFIGURAZIONE VARIE 
# IL TEST FINALE è QUELLO DI CANCELLARE TUTTO IL DATABASE E RICREARE E RIFARE LE QUERY E TUTTO, VIA UI 
# SI PUò ANCHE PENSARE DI SCRIVERE DEI TEST AUTOMATICI DEL CODICE


if __name__ == "__main__":

    # testing on the fly
    current_path = Path(__file__).resolve().parent

    # remove what's old
    client = QdrantClient(url="http://localhost:6333")
    if client.collection_exists("war"): client.delete_collection("war")
    del client

    hkb = HybridKnowledgeBuilder(collection_name="war")

    # DO NOT REPEAT INGESTION
    # hkb.ingest(current_path / "raw_data" / "gm2_and_gf.txt")

    # qeurying
    from qdrant_client.models import Prefetch, Document as QDocument, Fusion, FusionQuery
    from qdrant_ingestion import HybridKnowledgeBuilder

    hkb = HybridKnowledgeBuilder("war")
    query = "When were atomic bombs thrown on Japan? And how many?"
    results = hkb.client.query_points(
        collection_name=hkb.collection_name,
        prefetch=[
            Prefetch(
                query=QDocument(
                    text=query,
                    model=hkb.dense_embedding
                ),
                using="dense_vector",
                limit=5
            ),
            Prefetch(
                query=QDocument(
                    text=query,
                    model=hkb.sparse_embedding
                ),
                using="bm25_sparse_vector",
                limit=5
            )
        ], # score here is COSINE similarity
        query=FusionQuery(fusion=Fusion.RRF),   # score here is ranking
        limit=3,
        with_payload=True
    )
    import pprint
    pprint.pprint(results.points)

    passages = [
        {
            "id": response.id,
            "text": response.payload["page_content"],
            "meta": response.payload["meta"] # this is a dictionary
        }
        for response in results.points
    ]
    pprint.pprint(passages)
