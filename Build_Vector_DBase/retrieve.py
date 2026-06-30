import numpy as np
from collections import defaultdict

##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
persist_directory = r"C:\AI Stuff\CV_Matching_AI\VectorStore"
collection_name = "talent_matcher"
##-------Define functions to rank and retrieve best match--------
##------------------------------------------------------------------------

def compute_dynamic_k(collection, min_k=10, max_k=50, multiplier=3):
    #Computes a dynamic k based on the quantity of the CV in the DB
    #Returns number of chunks to retreive and number of detected CVs

    # Obtain metadata from Chroma collection
    all_metas = collection.get(include=["metadatas"])["metadatas"]

    # Extract unique names from the CV
    all_sources = {meta["source"] for meta in all_metas}
    total_cvs = len(all_sources)

    # Dynamic calculation of k
    k = min(max(total_cvs * multiplier, min_k), max_k)

    return k, total_cvs

def ranked_cvs(collection, query_embedding):
    # Retrieves data from vector data base
    # Compute dynamic k
    k, total_cvs = compute_dynamic_k(collection)

    #Search top CVs query FAISS
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "source": meta["source"],
            "content": doc,
            "distance": float(dist)
        })

    cv_scores = []
    #Group by CV
    grouped = defaultdict(list)
    for c in chunks:
        grouped[c["source"]].append(c)

    # Rank/Score per CV
    for cv_name, cv_chunks in grouped.items():
        distances = [c["distance"] for c in cv_chunks]
        # --- Convertir distancia FAISS → similitud coseno ---
        similarities = [1 - d for d in distances]

        #smallest final score should be the most similar vector
        #final_score = 0.7 * max(scores) + 0.3 * np.mean(scores)
        final_score = 0.7 * min(distances) + 0.3 * np.mean(distances)
        final_similarity = np.mean(similarities)
        match_score = round(final_similarity * 100, 2)

        cv_scores.append({
            "cv_name": cv_name,
            "final_similarity": final_similarity,
            "match_score": match_score,
            "max_score": max(distances),
            "mean_score": np.mean(distances),
            "chunks": cv_chunks,


        })

        # --- Si no hay CVs, evitar crash ---
        if len(cv_scores) == 0:
            return {
                "best_cv": None,
                "best_final_similarity": 0.0,
                "match_score": 0.0,
                "cv_scores": [],
                "context": "",
                "k_used": k,
                "total_cvs": total_cvs
            }

    #Order CV by most relevant
    #cv_scores = sorted(cv_scores, key=lambda x: x["final_score"])
    # Ordenar por mayor similitud
    cv_scores = sorted(cv_scores, key=lambda x: x["final_similarity"], reverse=True)

    #Getthe best CV name
    best = cv_scores[0]

    #Concatenate the text of the best CV
    best_chunks_sorted = sorted(best["chunks"], key=lambda x: x["distance"])


    # Organize chunks by relevance
    ordered = sorted(best["chunks"], key=lambda x: x["distance"])
    #Take only the best 3-5
    top_chunks = ordered[:5]
    #Build clean context
    context = "\n".join([c["content"].strip() for c in top_chunks])

    ranked_cv = {
        "best_cv": best["cv_name"],
        "best_final_similarity": best["final_similarity"],
        "match_score": best["match_score"],
        "ranked_cvs": [cv["cv_name"] for cv in cv_scores],
        "cv_scores": cv_scores,
        "context": context,
        "k_used": k,
        "total_cvs": total_cvs
    }

    return ranked_cv

##----DEBUG/TEST----------------------------------------------------------------------
def main():
    from Build_Vector_DBase import get_vector_store, embed_text
    print("\n=== TEST: Ranking CVs ===")
    query = input("Enter job description or query: ")

    collection = get_vector_store(persist_directory, collection_name)
    query_embedding = embed_text(query)

    results = ranked_cvs(collection, query_embedding)

    print("\nK usado:", results["k_used"])
    print("Total CVs:", results["total_cvs"])
    print("CVs rankeados:", results["ranked_cvs"])
    print("Scores:", results["cv_scores"])
    print("Mejor CV:", results["best_cv"])

if __name__ == "__main__":
    main()