import numpy as np
from collections import defaultdict
from Core.config_loader import config
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
persist_directory = r"C:\AI Stuff\CV_Matching_AI\VectorStore"
collection_name = "talent_matcher"
##-------Define functions to rank and retrieve best match--------
##------------------------------------------------------------------------
def convert_distance_to_score(distance: float) -> float:
    score = max(0, (1 - distance)**2 * 100)

    return round(score, 2)

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
            "score": float(dist)
        })

    #Group by CV
    grouped = defaultdict(list)
    for c in chunks:
        grouped[c["source"]].append(c)

    # Rank/Score per CV
    cv_scores = []
    for cv_name, cv_chunks in grouped.items():
        scores = [c["score"] for c in cv_chunks]

        #smallest final score should be the most similar vector
        #final_score = 0.7 * max(scores) + 0.3 * np.mean(scores)
        final_score = 0.7 * min(scores) + 0.3 * np.mean(scores)

        cv_scores.append({
            "cv_name": cv_name,
            "final_score": final_score,
            "max_score": max(scores),
            "mean_score": np.mean(scores),
            "chunks": cv_chunks,
            "match_score": convert_distance_to_score(final_score)
        })

    #Order CV by most relevant
    cv_scores = sorted(cv_scores, key=lambda x: x["final_score"])

    #Getthe best CV name
    best = cv_scores[0]
    best_cv = best["cv_name"]
    best_final_score = best["final_score"]
    match_score = convert_distance_to_score(best_final_score)

    #Concatenate the text of the best CV
    best_chunks_sorted = sorted(best["chunks"], key=lambda x: x["score"])
    #context = "\n\n".join([c["content"] for c in best_chunks_sorted])

    # Organize chunks by relevance
    ordered = sorted(best["chunks"], key=lambda x: x["score"])
    #Take only the best 3-5
    top_chunks = ordered[:5]
    #Build clean context
    context = "\n".join([c["content"].strip() for c in top_chunks])

    ranked_cv = {
        "best_cv": best_cv,
        "best_final_score" : best_final_score,
        "match_score": match_score,
        "ranked_cvs": [cv["cv_name"] for cv in cv_scores],
        "cv_scores": cv_scores,
        "context": context,
        "k_used": k,
        "total_cvs": total_cvs
    }

    return ranked_cv

##----DEBUG/TEST----------------------------------------------------------------------
#Test 0: El re‑ranking debe favorecer CVs con varios chunks relevantes, no solo uno.
#query = "Necesito un perfil con experiencia en migración a la nube, KPIs operativos y proyectos corporativos de analítica."
#Test 1 :  El sistema debe evaluar varios chunks por CV. No debe ganar un CV solo porque un chunk coincidió.
#query = "Busco un perfil con habilidades blandas fuertes, comunicación efectiva y liderazgo."
#Test 2 :
#query= "Busco alguien que sepa Python, APIs REST, Docker y tenga experiencia en desarrollo de software. Puede ser Full Stack, QA Automation o Data Engineer."
#Test 3: Debe ganar ese perfil en especificonked
#query= "Necesito un QA Automation Junior con epxeriencia en Wizeline"
#Pruebas desde input del usuario
#query = input("Enter question: ")
#collection = get_vector_store(persist_directory, collection_name)
#query_embedding = embed_text(query)
#results = ranked_cvs(collection, query_embedding)
#print("K usado:", results["k_used"])
#print("Total CVs:", results["total_cvs"])
#print("CVs rankeados:", results["ranked_cvs"])
#print("Scores:", results["cv_scores"])
#print("Mejor CV:", results["best_cv"])