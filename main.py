import os
from Feed_and_processing import (convert_pdf_to_json, convert_png_to_json, clean_json)
from Chunking import json_chunks
from Embeddings import (json_embeddings,to_vectors)
from Build_Vector_DBase import (get_vector_store, reset_vector_store, insert_and_index_chunks)
from Job_Matching import match_job_description
from Core.config_loader import config

#Base paths
BASE = r"C:\AI Stuff\CV_Matching_AI\Data"
ROOT = os.path.dirname(os.path.abspath(__file__))

CV_PATH = os.path.join(BASE, "CV", "CV_Test")
RAW_JSON = os.path.join(BASE, "Raw_Json")
CLEAN_JSON = os.path.join(BASE, "Clean_Json")
CHUNK_JSON = os.path.join(BASE, "Chunk_Json")
EMBED_JSON = os.path.join(BASE, "Embedding_Json")
VECTORSTORE = os.path.join(ROOT, "Vector_Store")
LOG_PATH = os.path.join(BASE, "System_Log")

chunk_size = 100
overlap = 20
active_profile = config["LLM"]["active_profile"]
section = f"LLM_{active_profile}"
collection_name = "talent_matcher"
reset=False




def main():

    print("\n=== 1) Extrayendo PDFs ===")
    convert_pdf_to_json(CV_PATH, RAW_JSON)

    print("\n=== 2) Extrayendo PNG/JPG ===")
    convert_png_to_json(CV_PATH, RAW_JSON)

    print("\n=== 3) Limpiando JSON ===")
    clean_json(RAW_JSON, CLEAN_JSON)

    print("\n=== 4) Chunking ===")
    json_chunks(CHUNK_JSON, CLEAN_JSON, chunk_size, overlap, show_results=False)

    print("\n=== 5) Generando embeddings ===")
    json_embeddings(EMBED_JSON, CHUNK_JSON)

    print("\n=== 6) Construyendo Vector Store ===")
    collection = get_vector_store(VECTORSTORE, "talent_matcher")
    collection = reset_vector_store(VECTORSTORE, "talent_matcher",reset)
    insert_and_index_chunks(EMBED_JSON, collection)

    print("\n=== 7) Matching ===")
    job_text = input("Ingresa la descripción de la vacante:\n\n")
    resultado = match_job_description(job_text,VECTORSTORE,collection_name, LOG_PATH,
                                      debug=False, report=True)

    print("\n=== Mejor CV ===")
    print(resultado["best_cv"])

    print("\n=== Explicación del LLM ===")
    print(resultado["explanation"])

    print("\n=== Tiempo de respuesta ===")
    print(resultado["llm_response_time"], "seconds")


if __name__ == "__main__":
    main()
