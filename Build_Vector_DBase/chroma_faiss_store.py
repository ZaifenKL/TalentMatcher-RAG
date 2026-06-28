import os
import json
from chromadb import PersistentClient
from collections import Counter
from Core.config_loader import config

##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
persist_directory = r"C:\AI Stuff\CV_Matching_AI\VectorStore"
collection_name = "talent_matcher"
embedding_json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Embedding_Json"

##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def reset_vector_store(persist_directory, collection_name):
    ##Deelete the existing collection and create new empty one
    client = PersistentClient(path=persist_directory)

    # Delete previous collection
    try:
        client.delete_collection(collection_name)
        print(f"[OK] Collection '{collection_name}' deleted...")
    except:
        print(f"[INFO] Previous collection '{collection_name}' did not exist.")

    # Create clean collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None
    )

    print(f"[OK] Collection '{collection_name}' created.")
    return collection

def verify_vector_store(collection):
    print("\n===== VERIFYING VECTOR STORE =====")

    # Total vector count
    total = collection.count()
    print(f"[INFO] ==== Total embeddings: {total}")

    data = collection.get(include=["metadatas", "documents", "embeddings"])

    metas = data["metadatas"]
    docs = data["documents"]
    embeds = data["embeddings"]
    ids = data["ids"]

    # Total unique files
    sources = [m["source"] for m in metas]
    unique_cvs = sorted(set(sources))
    print(f"[INFO] Total CVs: {len(unique_cvs)}")
    for cv in unique_cvs:
        print("   -", cv)

    # Number of chunks per CV
    print("\n[INFO] === Chunks per CV:")
    counter = Counter(sources)
    for cv, n in counter.items():
        print(f"   {cv}: {n} chunks")

    # Duplicated IDs
    print("\n[INFO] Verifying duplicated IDs...")
    dup_ids = [item for item, count in Counter(ids).items() if count > 1]
    if dup_ids:
        print("[WARNING!] Duplicated ids found:")
        for d in dup_ids:
            print("   -", d)
    else:
        print("[OK] No duplicated IDs found.")

    # Empty files
    print("\n[INFO] Verifying empty files...")
    empty_docs = [i for i, d in enumerate(docs) if not d.strip()]
    if empty_docs:
        print("WARNING!] Empty files found :", empty_docs)
    else:
        print("[OK] No empty documents found.")

    # Missing medatdata
    print("\n[INFO] Verifying missing metadata...")
    bad_meta = [i for i, m in enumerate(metas) if "source" not in m]
    if bad_meta:
        print("[WARNING!] Missing Medatadata at indexes:", bad_meta)
    else:
        print("[OK] Every metadata has a 'source'.")

    # Corrupted embeddings
    print("\n[INFO] Verifying corrupted embeddings...")
    bad_embeds = [i for i, e in enumerate(embeds) if e is None or len(e) == 0]
    if bad_embeds:
        print("[WARNING!] Corrupted embeddings at indexes:", bad_embeds)
    else:
        print("[OK] All embeddings valid.")

    print("\n===== VERIFICATION COMPLETE =====\n")


def get_vector_store(persist_directory, collection_name):
    #Init vector store and create collection
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
#collection = get_vector_store(persist_directory,collection_name)
#We reset in case the vector store already exists and in this case since I had
#a bug with the IDs I needed to reset
#collection = reset_vector_store(persist_directory, collection_name)
#Then we insert the embedding metadat and text into vector store
#insert_and_index_chunks(embedding_json_path,collection,show_results=True)
#Verifying that it ACTUALLY created the vector database
#verify_vector_store(collection)