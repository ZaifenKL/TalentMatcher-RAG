import gradio as gr
from Job_Matching import match_job_description

def run_match(job_text):
    #Execute motor
    result = match_job_description(job_text, log_path="", debug=False)

    best = result["best_cv"]
    explanation = result["explanation"]
    top5 = "\n".join([cv["cv_name"] for cv in result["cv_scores"][:5]])

    ranking_table = "RANK | CV | SCORE\n"
    ranking_table += "----------------------\n"
    for idx, cv in enumerate(result["cv_scores"], start=1):
        ranking_table += f"{idx} | {cv['cv_name']} | {cv['final_score']:.4f}\n"

    return best, top5, explanation, ranking_table

with gr.Blocks() as demo:
    gr.Markdown("# Talent Matcher — UI Demo")
    gr.Markdown("Ingresa una descripción de vacante para obtener el mejor CV.")

    job_input = gr.Textbox(label="Descripción de la vacante", lines=4)

    btn = gr.Button("Analizar")

    best_output = gr.Textbox(label="Mejor CV")
    top5_output = gr.Textbox(label="Top 5 CVs")
    explanation_output = gr.Textbox(label="Explicación del LLM", lines=10)
    ranking_output = gr.Textbox(label="Ranking completo", lines=15)

    btn.click(
        run_match,
        inputs=job_input,
        outputs=[best_output, top5_output, explanation_output, ranking_output]
    )

demo.launch()
