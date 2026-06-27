import ollama
from Core.config_loader import config
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
##------------------------------------------------------------------------
active_profile = config["LLM"]["active_profile"]
section = f"LLM_{active_profile}"
MODEL = config[section]["model"]
TEMPERATURE = float(config[section]["temperature"])
MAX_TOKENS = int(config[section]["max_tokens"])
##-------Define functions to optimize the pdf extraction data flow--------
##-----------------------------------------------------------------------
def run_llm(prompt: str):
    # Run the configured model (phi or llama) using Ollama
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS
            }
        )
        return response["message"]["content"]

    except Exception as e:
        return f"[ERROR LLM] {str(e)}"


def explain_match(query: str, results: dict) -> str:
    #Generates and explanation on why this CV matches the job description
    context = results["context"]
    best_cv = results["best_cv"]

    prompt = f"""
    El usuario busca: {query}

    El CV más relevante es: {best_cv}

    Fragmentos relevantes del CV:
    {context}

    Explica por qué este CV es el mejor match.
    Resume las habilidades clave.
    Evalúa el match en una escala del 0 al 100.
    Actúa como si fueras un profesional experto en búsqueda de talento.
    """

    return run_llm(prompt)
