from Build_Vector_DBase import (embed_text, get_vector_store, ranked_cvs)
from LLM_Engine import explain_match, test_llm_connection
from datetime import datetime
from Report_Builder import (build_ranking_report,export_markdown_report)
import os

# ------------------------------------------------------------------------
# ------------------ Constant Values -------------------------------------
# ------------------------------------------------------------------------
persist_directory = r"C:\AI Stuff\CV_Matching_AI\VectorStore"
collection_name = "talent_matcher"
log_path = r"C:\AI Stuff\CV_Matching_AI\Data\System_Log"
# ------------------------------------------------------------------------
# ------------------ Main Matching Function -------------------------------
# ------------------------------------------------------------------------

def match_job_description(job_text: str,
                          persist_directory: str,
                          collection_name: str,
                          log_path: str,
                          debug=False,
                          report=False):
    #Input the description of the job opening and returns the best CV, complete ranking, LLM explanation and
    # markdown format report if debug is TRUE

    if debug:
        print("\n=== Testing LLM Connection ===")
        test_llm_connection()

    # 1. Embedding de la vacante
    job_embedding = embed_text(job_text)

    # 2. Cargar Vector Store
    collection = get_vector_store(persist_directory, collection_name)

    # 3. Ranking de CVs
    ranking = ranked_cvs(collection, job_embedding)

    # 4. Explicación del LLM
    explanation, llm_response_time, prompt = explain_match(job_text, ranking)

    if report:
        # 5. Construir reporte
        report = build_ranking_report(ranking, explanation, llm_response_time,job_text,prompt)
        #timestamp for unique report name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_path, f"reporte_match_{timestamp}.md")
        export_markdown_report(report, log_file)

    # 7. Retornar datos útiles
    results = {
        "best_cv": ranking["best_cv"],
        "best_final_similarity": ranking["best_final_similarity"],
        "match_distance": ranking["match_distance"],
        "ranked_cvs": ranking["ranked_cvs"],
        "cv_scores": ranking["cv_scores"],
        "context": ranking["context"],
        "explanation": explanation,
        "llm_response_time": llm_response_time,
        "prompt": prompt
    }

    return results

# ------------------------------------------------------------------------
# ------------------ DEBUG / TEST ----------------------------------------
# ------------------------------------------------------------------------
def main():
    job_text = input("Please enter your job description: ")
    resultado = match_job_description(job_text, persist_directory, collection_name,
                                      log_path,
                                      debug=False, report=True,
                                      )

    print("\n=== Explanation del LLM ===")
    print(resultado["explanation"])
    print(f"{resultado['llm_response_time']} seconds")

if __name__ == "__main__":
    main()