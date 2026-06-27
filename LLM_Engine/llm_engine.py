import ollama
from LLM_Engine.config_loader import config

##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
##------------------------------------------------------------------------
active_profile = config["LLM"]["active_profile"]
section = f"LLM_{active_profile}"

# ============================
# 2. Cargar parámetros del modelo
# ============================
MODEL = config[section]["model"]
TEMPERATURE = float(config[section]["temperature"])
MAX_TOKENS = int(config[section]["max_tokens"])

# ============================
# 3. Función genérica para llamar al LLM
# ============================
def run_llm(prompt: str) -> str:
    """
    Ejecuta el modelo seleccionado (phi o llama) usando Ollama.
    """
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

# ============================
# 4. Función especializada para explicar el match
# ============================
def explain_match(query: str, results: dict) -> str:
    """
    Genera una explicación del match entre el query y el CV.
    """
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
    """

    return run_llm(prompt)
