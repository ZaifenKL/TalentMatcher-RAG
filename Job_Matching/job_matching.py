from Build_Vector_DBase import (embed_text, get_vector_store, ranked_cvs)
from LLM_Engine import explain_match, test_llm_connection
import time
import os

# ------------------------------------------------------------------------
# ------------------ Constant Values -------------------------------------
# ------------------------------------------------------------------------
persist_directory = r"C:\AI Stuff\CV_Matching_AI\VectorStore"
collection_name = "talent_matcher"

# ------------------------------------------------------------------------
# ------------------ Main Matching Function -------------------------------
# ------------------------------------------------------------------------

def match_job_description(job_text: str, log_file: str, debug=False):
    """
    Recibe una descripción de vacante y devuelve:
    - Mejor CV
    - Ranking completo
    - Explicación del LLM
    - Exporta un reporte en Markdown si debug=True
    """

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

    # 5. Construir reporte
    report = build_ranking_report(ranking, explanation, llm_response_time)

    # 6. Exportar reporte si debug=True
    if debug:
        print(f"\nLLM Response Time: {llm_response_time}")
        print(report["ranking_table"])
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


# ------------------------------------------------------------------------
# ------------------ Report Builder --------------------------------------
# ------------------------------------------------------------------------

def build_ranking_report(ranking: dict, llm_response: str, llm_response_time: float) -> dict:
    """
    Construye un reporte estructurado del ranking de CVs.
    """

    # Construir tabla
    table_lines = []
    table_lines.append("RANK | CV | SCORE | MAX | MEAN")
    table_lines.append("-------------------------------------------")

    for idx, cv in enumerate(ranking["cv_scores"], start=1):
        table_lines.append(
            f"{idx} | {cv['cv_name']} | {cv['final_score']:.4f} | "
            f"{cv['max_score']:.4f} | {cv['mean_score']:.4f}"
        )

    table_text = "\n".join(table_lines)

    # Reporte final
    report = {
        "best_cv": ranking["best_cv"],
        "ranking_table": table_text,
        "ranking_json": ranking["cv_scores"],
        "llm_explanation": llm_response,
        "llm_response_time_seconds": round(llm_response_time, 4)
    }

    return report


# ------------------------------------------------------------------------
# ------------------ Markdown Export -------------------------------------
# ------------------------------------------------------------------------

def export_markdown_report(report: dict, log_file: str) -> str:
    """
    Exporta un archivo Markdown con el reporte del sistema.
    """

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    md = []
    md.append("# Reporte de Matching de CVs\n")
    md.append(f"**Mejor CV:** `{report['best_cv']}`\n")
    md.append("---\n")

    md.append("## Ranking de CVs\n")
    md.append("```\n" + report["ranking_table"] + "\n```\n")

    md.append("## Explicación del Modelo\n")
    md.append(report["llm_explanation"] + "\n")

    md.append("## Tiempo de respuesta del LLM\n")
    md.append(f"- `{report['llm_response_time_seconds']} segundos`\n")

    md.append("---\n")
    md.append("Generado automáticamente por el sistema de Talent Matcher.\n")

    markdown_text = "\n".join(md)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    return f"Reporte guardado en: {log_file}"


# ------------------------------------------------------------------------
# ------------------ DEBUG / TEST ----------------------------------------
# ------------------------------------------------------------------------

if __name__ == "__main__":
    log_file = r"C:\AI Stuff\CV_Matching_AI\Data\System_Log\reporte.md"

    resultado = match_job_description(
        "Busco un QA Automation Junior con experiencia en Wizeline",
        log_file,
        debug=True
    )

    print("\n=== Explanation del LLM ===")
    print(resultado["explanation"])
