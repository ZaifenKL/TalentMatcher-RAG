from openai import OpenAI
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
client = OpenAI()
#query_embedding es un vector the consulta
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def embed_text(text):
    #Converts user question/interaction into embedding
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding