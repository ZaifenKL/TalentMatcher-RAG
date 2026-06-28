from sympy.physics.units import seconds

from Build_Vector_DBase import (embed_text, get_vector_store, ranked_cvs)
from LLM_Engine import explain_match, test_llm_connection
from datetime import datetime
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

def match_job_description(job_text: str, log_path: str, debug=False, report=False):
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
    explanation, llm_response_time = explain_match(job_text, ranking)

    if report:
        # 5. Construir reporte
        report = build_ranking_report(ranking, explanation, llm_response_time,job_text)
        # 6. Exportar reporte si debug=True
        #timestamp for unique report name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_path, f"reporte_match_{timestamp}.md")
        export_markdown_report(report, log_file)

    # 7. Retornar datos útiles
    results = {
        "best_cv": ranking["best_cv"],
        "ranked_cvs": ranking["ranked_cvs"],
        "cv_scores": ranking["cv_scores"],
        "context": ranking["context"],
        "explanation": explanation,
        "llm_response_time": llm_response_time
    }

    return results

def build_ranking_report(ranking: dict, llm_response: str, llm_response_time: float, job_text) -> dict:
    #Build table
    table_lines = []
    table_lines.append("RANK | CV | SCORE | MAX | MEAN")
    table_lines.append("-------------------------------------------")

    for idx, cv in enumerate(ranking["cv_scores"], start=1):
        table_lines.append(
            f"{idx} | {cv['cv_name']} | {cv['final_score']:.4f} | "
            f"{cv['max_score']:.4f} | {cv['mean_score']:.4f}"
        )

    table_text = "\n".join(table_lines)

    # Final report
    report = {

        "best_cv": ranking["best_cv"],
        "ranking_table": table_text,
        "ranking_json": ranking["cv_scores"],
        "job_text": job_text,
        "llm_explanation": llm_response,
        "llm_response_time_seconds": round(llm_response_time, 4)
    }

    return report

def export_markdown_report(report: dict, log_file: str) -> str:
    #Exports system log to markdown format
    #Create folder if it does not exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append("# Report : Matching CVs\n")
    md.append(f"**Mejor CV:** `{report['best_cv']}`\n")

    md.append("## Report generation datetime")
    md.append(f"- `{timestamp}`\n")
    md.append("---\n")
    md.append("---\n")

    md.append("## CVs Ranking\n")
    md.append("```\n" + report["ranking_table"] + "\n```\n")

    md.append("## User Input\n")
    md.append("```\n" + report["job_text"] + "\n```\n")

    md.append("## Model Explanation\n")
    md.append(report["llm_explanation"] + "\n")

    md.append("## LLM Response time\n")
    md.append(f"- `{report['llm_response_time_seconds']} seconds`\n")

    markdown_text = "\n".join(md)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    return f"System log saved: {log_file}"


# ------------------------------------------------------------------------
# ------------------ DEBUG / TEST ----------------------------------------
# ------------------------------------------------------------------------
resultado = match_job_description("Busco un QA Automation Junior con experiencia en Wizeline",
        log_path,
        debug=False , report=True)

print("\n=== Explanation del LLM ===")
print(resultado["explanation"])
print(f"{resultado["llm_response_time"]} seconds")
