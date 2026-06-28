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
    best_final_score = results["best_final_score"]

    prompt = f"""
    INSTRUCCIONES DEL SISTEMA (NO MOSTRAR AL USUARIO):
    La primera línea DEBE ser exactamente: "CV seleccionado: {best_cv}".
    No inventes nombres de CV.
    Usa únicamente la información del CV seleccionado.
    Distancia interna del mejor CV: {best_final_score}.
    Reglas de score:
    - Si distancia > 0.50 → score < 40.
    - Si distancia > 0.70 → score < 20.
    - Si distancia > 0.90 → score < 5.
    No menciones distancias ni conceptos técnicos.

    VACANTE:
    {query}

    FRAGMENTOS DEL CV:
    {context}

    TAREAS:
    Como reclutador técnico experto:
    - Explica por qué este CV fue seleccionado.
    - Resume las habilidades relevantes.
    - Evalúa el match del 0 al 100 siguiendo las reglas internas.
    - Si el match es bajo, explica por qué.
    - No inventes habilidades que no estén en el CV.
    """

    start = time.time()
    explanation = run_llm(prompt)
    llm_response_time = round(time.time() - start, 4)

    return explanation, llm_response_time, prompt

##----DEBUG/TEST----------------------------------------------------------------------
#respuesta = test_llm_connection()
