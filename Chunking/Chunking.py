from transformers import AutoTokenizer
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
##------------------------------------------------------------------------
chunk_size = 250
overlap =50
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def chunk_text_hf(text, chunk_size, overlap):
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        # Convert tokens to text
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move window with overlap
        start += chunk_size - overlap

    return chunks

def json_chunks()

##----DEBUG/TEST------------------------------------------------------------------

