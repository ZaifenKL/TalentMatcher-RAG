from sentence_transformers import SentenceTransformer
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
#Load Model
#Load model a single time
embedding_model = SentenceTransformer(r"C:\Users\User\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2")

embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def embed_text(text):
    embedding = model.encode(text)
    return embedding.tolist()
##----DEBUG/TEST---------------------------------------------------------------
text = "Hola podria buscar un ingeniero con 3 años de experiencia en Python?"
embed_text = embed_text(text)
print(f"This is my embedding : {embed_text}")