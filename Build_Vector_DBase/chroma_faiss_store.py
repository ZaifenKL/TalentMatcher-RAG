import os
import json
from chromadb import PersistentClient

##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
persist_directory = r"C:\AI Stuff\CV_Matching_AI\VectorStore"
collection_name = "talent_matcher"
embedding_json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Embedding_Json"

##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------


def get_vector_store(persist_directory, collection_name):
    #Init vector store ans create collection
    client = PersistentClient(path=persist_directory)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        #This fucntion activates faiss
        embedding_function=None
    )

    return collection

def add_embeddings(collection, chunks, source_name):
    #Insert chunks (text, embeddings and metadata) into de chroma collection
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["content"] for c in chunks],
        metadatas=[{
            "source": source_name,
            "start_token": c["start_token"],
            "end_token": c["end_token"]
        } for c in chunks]
    )

def insert_and_index_chunks(embedding_json_path,collection,show_results=False):
    for root, dirs, files in os.walk(embedding_json_path):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        data = json.load(f)

                        chunks = data["chunks"]
                        source_name = data["name"]
                        add_embeddings(collection, chunks, source_name)

                        if show_results:
                            print(f"=====Embeddings added from: {file}")

                except Exception as e:
                    print(f"Error en {file}: {e}")

##----DEBUG/TEST----------------------------------------------------------------------
#First we run this function to initialize vector store and create the collection
collection = get_vector_store(persist_directory,collection_name)
#Then we insert the embedding metadat and text into vector store
insert_and_index_chunks(embedding_json_path,collection,show_results=True)