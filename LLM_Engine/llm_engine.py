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
    match_score = results["match_score"]

    prompt = f"""
    INSTRUCCIONES DEL SISTEMA:
    La primera línea DEBE ser exactamente: "CV seleccionado: {best_cv}".
    No inventes nombres de CV.
    Usa únicamente la información del CV seleccionado.

    VACANTE:
    {query}

    FRAGMENTOS DEL CV:
    {context}
    
    SCORE FINAL DEL MATCH: 
    {match_score}/100

    TAREAS:
    Como reclutador técnico experto:
    - Resume las habilidades relevantes y experiencia que aparecen en el CV.
    - Usa el score proporcionado como indicador final del match.
    - Si el score es bajo, explica por qué el perfil no encaja.
    - Si el score es alto, explica por qué el perfil encaja.
    - No inventes habilidades, proyectos, certificaciones ni experiencia.
    """

    start = time.time()
    explanation = run_llm(prompt)
    llm_response_time = round(time.time() - start, 4)

    return explanation, llm_response_time, prompt

##----DEBUG/TEST----------------------------------------------------------------------
def main():
    print("\n=== LLM Debug Test ===")
    test_llm_connection()
    print("\n=== LLM Test Complete ===")


if __name__ == "__main__":
    main()
