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

    matching_data = {"best_cv": best_cv,
        "ranked_cvs": [cv["cv_name"] for cv in cv_scores],
        "cv_scores": cv_scores,
        "context": context_text}

    # 10. Return everything
    return matching_data
##----DEBUG/TEST----------------------------------------------------------------------
#Test 0: El re‑ranking debe favorecer CVs con varios chunks relevantes, no solo uno.
#query = "Necesito un perfil con experiencia en migración a la nube, KPIs operativos y proyectos corporativos de analítica."
#Test 1 :  El sistema debe evaluar varios chunks por CV. No debe ganar un CV solo porque un chunk coincidió.
#query = "Busco un perfil con habilidades blandas fuertes, comunicación efectiva y liderazgo."
#Test 2 :
#query= "Busco alguien que sepa Python, APIs REST, Docker y tenga experiencia en desarrollo de software. Puede ser Full Stack, QA Automation o Data Engineer."
#Test 3: Debe ganar ese perfil en especifico
query= "Necesito un QA Automation Junior con epxeriencia en Wizeline"
#Pruebas desde input del usuario
#query = input("Enter question: ")
results = retrieve_context(query)
print("===== Ranked CVs (best first) =====")
for cv in results["ranked_cvs"]:
    print(cv)
print("\n===== CV Scores =====")
for cv in results["cv_scores"]:
    print(f"{cv['cv_name']}  |  final_score={cv['final_score']:.4f}")

print("====Process complete====")