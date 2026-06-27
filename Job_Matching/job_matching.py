from Build_Vector_DBase import (embed_text,get_vector_store,ranked_cvs)
from LLM_Engine import explain_match, test_llm_connection
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
persist_directory = r"C:\AI Stuff\CV_Matching_AI\VectorStore"
collection_name = "talent_matcher"

##-------Define functions to handle match job work flow--------
##------------------------------------------------------------------------

def match_job_description(job_text: str, debug=False):
    #Receives an explanation on the desired job profile and returns
    #The best CV fit, ranking and LLM explanation
    if debug:
        test_llm_connection()

    # 1. Job description
    job_embedding = embed_text(job_text)

    # 2. Load Vector Store
    collection = get_vector_store(persist_directory, collection_name)

    # 3. Ranking
    ranking = ranked_cvs(collection, job_embedding)

    # 4. LLM explanation on match
    explanation = explain_match(job_text, ranking)

    # 5. Join relevant data
    return {
        "best_cv": ranking["best_cv"],
        "ranked_cvs": ranking["ranked_cvs"],
        "cv_scores": ranking["cv_scores"],
        "context": ranking["context"],
        "explanation": explanation
    }

##----DEBUG/TEST----------------------------------------------------------------------
resultado = match_job_description("Busco un QA Automation Junior con experiencia en Wizeline", debug=True)
print(resultado)
print(f"Explanation del LLM: {resultado['explanation']}")
