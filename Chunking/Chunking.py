from transformers import AutoTokenizer
import json
import os
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
##------------------------------------------------------------------------
chunk_size = 100
overlap = 20
chunk_json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Chunk_Json"
clean_json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Clean_Json"
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def make_chunks(text, chunk_size, overlap):

    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = start + chunk_size
        #Do not cut words
        while end < len(tokens) and tokenizer.convert_ids_to_tokens(tokens[end]).startswith("##"):
            end += 1

        #Update tokens
        chunk_tokens = tokens[start:end]

        # Decode and clean subwords
        chunk_text = tokenizer.decode(chunk_tokens).replace("##", "")

        chunk_dict = {
            "id": f"chunk_{chunk_id}",
            "start_token": start,
            "end_token": min(end, len(tokens) - 1),
            "content": chunk_text
            }
        # Add element to the end of the list
        chunks.append(chunk_dict)

        # Move overlap interval
        start += chunk_size - overlap
        chunk_id += 1

    return chunks

def json_chunks(chunk_json_path, clean_json_path, chunk_size, overlap):
    for root, dirs, files in os.walk(clean_json_path):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        data = json.load(f)

                        chunks = []
                        chunks = make_chunks(data["text"]["content"], chunk_size, overlap)

                        # --- Build Clean JSON -------------
                        json_data = {
                            "name": data["name"],
                            "source_path": data["source_path"],
                            "source_format": data["source_format"],
                            "chunk_size": chunk_size,
                            "overlap": overlap,
                            "number_of_chunks": len(chunks),

                            "chunks": chunks
                        }

                        # --- Ensure folder exists ---
                        output_dir = os.path.join(chunk_json_path, data["source_format"])
                        os.makedirs(output_dir, exist_ok=True)

                        # --- Save JSON --------------
                        json_filename = os.path.splitext(file)[0] + ".json"
                        json_output = os.path.join(chunk_json_path, data["source_format"], json_filename)
                        with open(json_output, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)
                        print(f"====JSON saved: {json_filename}====")

                except Exception as e:
                    print(f"Error en {file}: {e}")

##----DEBUG/TEST------------------------------------------------------------------
json_chunks(chunk_json_path, clean_json_path, chunk_size, overlap)
