from Core.config_loader import config
from LLM_Engine.llm_engine import run_llm

print("=== Probando conexión con el LLM ===")
print("Perfil activo:", config["LLM"]["active_profile"])

prompt = "Hola, ¿puedes responder un mensaje corto para confirmar que estás funcionando?"
respuesta = run_llm(prompt)

print("\n=== Respuesta del modelo ===")
print(respuesta)
