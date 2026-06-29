from sentence_transformers import SentenceTransformer
import json
import os
from Core.config_loader import config

##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
chunk_json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Chunk_Json"
embedding_json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Embedding_Json"

#Load model a single time
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def embed_chunks(chunks):
    # Extract only the text
    texts = [chunk["content"] for chunk in chunks]

    # Generate embeddings
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Build new chunk list with embeddings
    embedding_results = []
    for chunk, emb in zip(chunks, embeddings):
        embedding_results.append({
            "id": chunk["id"],
            "start_token": chunk["start_token"],
            "end_token": chunk["end_token"],
            "content": chunk["content"],
            "embedding": emb.tolist()
        })
    return embedding_results

def json_embeddings(embedding_json_path, chunk_json_path,show_results=False):
    #Integrates embeddings into our json files
    for root, dirs, files in os.walk(chunk_json_path):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        data = json.load(f)

                        if show_results:
                            print(f"===Processing {file}===")
                        new_chunks = embed_chunks(data["chunks"])
                        if show_results:
                            print(f"This are the results of the embeddings: \n {new_chunks}")

                        # --- Build Clean JSON -------------
                        json_data = {

                            **data,
                            "chunks": new_chunks
                        }

                        # --- Ensure folder exists ---
                        output_dir = os.path.join(embedding_json_path, data["source_format"])
                        os.makedirs(output_dir, exist_ok=True)

                        # --- Save JSON --------------

                        json_filename = os.path.splitext(file)[0] + ".json"
                        json_output = os.path.join(embedding_json_path, data["source_format"], json_filename)
                        with open(json_output, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)
                        if show_results:
                            print(f"====JSON saved: {json_filename}====")

                except Exception as e:
                    print(f"Error en {file}: {e}")

##----DEBUG/TEST------------------------------------------------------------------
def main():
    print("\n=== Generating Embeddings for All Chunk JSON Files ===")
    json_embeddings(embedding_json_path, chunk_json_path, show_results=True)
    print("\n=== Embedding Generation Complete ===")

if __name__ == "__main__":
    main()