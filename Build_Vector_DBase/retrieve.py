import numpy as np
from collections import defaultdict
from chroma_faiss_store import get_vector_store
from embedder import embed_text
from vector_semantic_search import search_similar_chunks
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
persist_directory = r"C:\AI Stuff\CV_Matching_AI\VectorStore"
collection_name = "talent_matcher"
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def compute_dynamic_k(collection, min_k=10, max_k=50, multiplier=3):
    #Computes a dynamic k based on the quantity of the CV in the DB
    #Returns number of chunks to retreive and number of detected CVs

    # Obtain metadat from Chroma collection
    all_metas = collection.get(include=["metadatas"])["metadatas"]

    # Extract unique names from the CV
    all_sources = {meta["source"] for meta in all_metas}
    total_cvs = len(all_sources)

    # Dynamic calculation of k
    k = min(max(total_cvs * multiplier, min_k), max_k)

    return k, total_cvs

def retrieve_context(query):
    # 1. Load vector store
    collection = get_vector_store(persist_directory, collection_name)

    # 2. Embed query
    query_embedding = embed_text(query)

    #Dynamic calculation of k
    k, total_cvs = compute_dynamic_k(collection)

    # 3. Retrieve top-k chunks
    results = search_similar_chunks(collection, query_embedding, k)

    # 4. Convert results into a list of dicts
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "source": meta["source"],
            "content": doc,
            "score": float(dist)
        })

    # 5. Group chunks by CV
    grouped = defaultdict(list)
    for c in chunks:
        grouped[c["source"]].append(c)

    # 6. Compute ranking score per CV
    cv_scores = []
    for cv_name, cv_chunks in grouped.items():
        scores = [c["score"] for c in cv_chunks]

        max_score = max(scores)
        mean_score = np.mean(scores)

        # Weighted score (tuneable)
        final_score = 0.7 * max_score + 0.3 * mean_score

        cv_scores.append({
            "cv_name": cv_name,
            "max_score": max_score,
            "mean_score": mean_score,
            "final_score": final_score,
            "chunks": cv_chunks
        })

    # 7. Sort CVs by final score (ascending = más relevante)
    cv_scores = sorted(cv_scores, key=lambda x: x["final_score"])

    # 8. Best CV
    best_cv = cv_scores[0]["cv_name"]

    # 9. Concatenate context of best CV
    best_chunks_sorted = sorted(cv_scores[0]["chunks"], key=lambda x: x["score"])
    context_text = "\n\n".join([c["content"] for c in best_chunks_sorted])

    # 10. Return everything
    return {
        "best_cv": best_cv,
        "ranked_cvs": [cv["cv_name"] for cv in cv_scores],
        "cv_scores": cv_scores,
        "context": context_text
    }
##----DEBUG/TEST----------------------------------------------------------------------
#query = "Busco un ingeniero de datos con Python y Azure"
query = input("Enter question: ")
results = retrieve_context(query)
print(f"=====This are the most relevant documents:\n{results['best_cv']}")
print(f"=====Number of documents retrieved: {len(results)}")
print("====Process complete====")