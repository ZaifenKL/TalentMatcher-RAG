import ollama
from Core.config_loader import config
import time
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
##------------------------------------------------------------------------
active_profile = config["LLM"]["active_profile"]
section = f"LLM_{active_profile}"
MODEL = config[section]["model"]
TEMPERATURE = float(config[section]["temperature"])
MAX_TOKENS = int(config[section]["max_tokens"])
RESPONSE_STYLE = config[section]["response_style"]
##-------Define functions to optimize the pdf extraction data flow--------
##-----------------------------------------------------------------------
def test_llm_connection():
    #Debuggin function
    print("=== Testing LLM Connection ===")
    print("Active Profile:", config["LLM"]["active_profile"])

    prompt = "Hola, ¿puedes responder un mensaje corto para confirmar que estás funcionando?"
    respuesta = run_llm(prompt)

    print("\n===Model Response ===")
    print(respuesta)

    return respuesta

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
    # Read active profile
    #active_profile = config["LLM"]["active_profile"]
    #section = f"LLM_{active_profile}"
    #raw_style = config[section].get("response_style", "").strip()
    #RESPONSE_STYLE = raw_style if raw_style else ""
    #Generates and explanation on why this CV matches the job description
    context = results["context"]
    best_cv = results["best_cv"]


    prompt = f"""
    El usuario busca: {query}
    El CV más relevante identificado por el sistema es: {best_cv}
    Fragmentos relevantes del CV seleccionado:
    {context}
    Instrucciones:
    1. Menciona explícitamente el nombre del CV seleccionado al inicio de tu respuesta.
    2. Explica por qué este CV es el mejor match para la vacante.
    3. Resume las habilidades clave que coinciden con los requisitos.
    4. Evalúa el match en una escala del 0 al 100.
    5. Redacta como un profesional experto en reclutamiento técnico.
    """
    start = time.time()
    explanation = run_llm(prompt)
    llm_response_time = round(time.time() - start, 4)

    return explanation, llm_response_time

##----DEBUG/TEST----------------------------------------------------------------------
#respuesta = test_llm_connection()
