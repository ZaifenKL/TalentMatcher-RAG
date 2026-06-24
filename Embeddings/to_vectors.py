from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
chunk_json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Chunk_Json"
embedding_json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Embedding_Json"

#Load model a single time
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def embed_chunks(chunks):
    #Input the list of chunks and returns a list of embeddings (one per chunk)
    texts = [chunk["content"] for chunk in chunks]

    #Generate embedding
    embeddings = embedding_model.encode(texts, convert_to_numpy=True,
        normalize_embeddings=True
    )

    embedding_results = []
    for chunk, emb in zip(chunks, embeddings):
        embedding_results.append({
            "id": chunk["id"],
            "content": chunk["content"],
            "embedding": emb.tolist()  # JSON-friendly
        })

    return embedding_results

def json_embeddings(embedding_json_path, chunk_json_path):
    #Integrates embeddings into our json files
    for root, dirs, files in os.walk(chunk_json_path):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        data = json.load(f)

                        embeddings = []
                        embeddings = embed_chunks(data["chunks"])

                        # --- Build Clean JSON -------------
                        json_data = {

                            **data,
                            "embeddings": embeddings
                        }

                        # --- Ensure folder exists ---
                        output_dir = os.path.join(embedding_json_path, data["source_format"])
                        os.makedirs(output_dir, exist_ok=True)

                        # --- Save JSON --------------
                        json_filename = os.path.splitext(file)[0] + ".json"
                        json_output = os.path.join(embedding_json_path, data["source_format"], json_filename)
                        with open(json_output, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)
                        print(f"====JSON saved: {json_filename}====")

                except Exception as e:
                    print(f"Error en {file}: {e}")

##----DEBUG/TEST------------------------------------------------------------------
json_embeddings(embedding_json_path, chunk_json_path)