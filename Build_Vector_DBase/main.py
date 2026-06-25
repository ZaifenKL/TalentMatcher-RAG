from chroma_faiss_store import get_vector_store
from embedder import embed_text
from vector_semantic_search import search_similar_chunks
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
query_from_user = "I need a data engineer with expertise in Python and Azure"
k_chunks = 5
persist_directory = r"C:\AI Stuff\CV_Matching_AI\VectorStore"
collection_name = "talent_matcher"
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def main(query_from_user,k_chunks,persist_directory,collection_name):
    collection = get_vector_store(persist_directory,collection_name)

    query_embedding = embed_text(query_from_user)

    results = search_similar_chunks(collection, query_embedding, k_chunks)

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        print("Document:", meta["source"])
        print("Chunk:", doc[:120], "...")
        print("Distancia:", dist)
        print("-----")

if __name__ == "__main__":
    main(query_from_user, k_chunks,persist_directory,collection_name)