from datetime import datetime
import os

def build_ranking_report(ranking: dict, llm_response: str, llm_response_time: float, job_text, prompt) -> dict:
    #Build table
    table_lines = []
    table_lines.append("RANK | CV | SIMILARITY | MATCH_SCORE | MAX | MEAN | SCORE")
    table_lines.append("-------------------------------------------")

    for idx, cv in enumerate(ranking["cv_scores"], start=1):
        table_lines.append(

            f"{idx} | {cv['cv_name']} | "
            f"{cv['final_similarity']:.4f} | "
            f"{cv['match_distance']}/100 | "
            f"{cv['max_distance']:.4f} | "
            f"{cv['mean_distance']:.4f}"
        )

    table_text = "\n".join(table_lines)

    # Final report
    report = {

        "best_cv": ranking["best_cv"],
        "ranking_table": table_text,
        "ranking_json": ranking["cv_scores"],
        "job_text": job_text,
        "llm_prompt" : prompt,
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

    md.append("## Base Prompt\n")
    md.append(report["llm_prompt"] + "\n")

    markdown_text = "\n".join(md)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    return f"System log saved: {log_file}"