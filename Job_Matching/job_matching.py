from Build_Vector_DBase.embedder import embed_text
from chroma_faiss_store import get_vector_store
from Ranking.ranked_cvs import ranked_cvs
from LLM_Engine.llm_engine import explain_match


def match_job_description(job_text: str):
    """
    Recibe una descripción de vacante y devuelve:
    - Mejor CV
    - Ranking completo
    - Explicación del LLM
    """

    # 1) Embedding de la vacante
    job_embedding = embed_text(job_text)

    # 2) Cargar vector store
    collection = get_vector_store(persist_directory, collection_name)

    # 3) Ranking usando tu motor existente
    ranking = ranked_cvs(collection, job_embedding)

    # 4) Explicación del LLM
    explanation = explain_match(job_text, ranking)

    # 5) Empaquetar todo
    return {
        "best_cv": ranking["best_cv"],
        "ranked_cvs": ranking["ranked_cvs"],
        "cv_scores": ranking["cv_scores"],
        "context": ranking["context"],
        "explanation": explanation
    }