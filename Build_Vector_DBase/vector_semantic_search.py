
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
k= 5
#query_embedding es un vector the consulta
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def search_similar_chunks(collection, query_embedding,k):
    #This is the semantic search function based on vectors, it searches for the most similar chunks using coseno
    #returns a dictionary
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results

##----DEBUG/TEST----------------------------------------------------------------------