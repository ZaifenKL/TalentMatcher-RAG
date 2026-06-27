import gradio as gr
# Import pipeline
from Build_Vector_DBase.embedder import embed_text
from Build_Vector_DBase.retrieve import ranked_cvs
from Build_Vector_DBase.chroma_faiss_store import load_collection
#from LLM_Engine.llm_explainer import explain_match

# Load collection
collection = load_collection()

def match(query):
    # 1. Embedding
    query_embedding = embed_text(query)

    # 2. Ranking
    results = ranked_cvs(collection, query_embedding)

    # 3. Explicación del LLM
    #explanation = explain_match(query, results)

    # 4. Armar salida bonita
    output = f"""
### 🧠 Mejor CV:
**{results['best_cv']}**

---

### 📊 Ranking completo:
{results['ranking']}

---

### 📝 Explicación del LLM:
{explanation}

---

### 🔍 Fragmentos relevantes:
{results['context']}
"""

    return output


# Interfaz Gradio
ui = gr.Interface(
    fn=match,
    inputs=gr.Textbox(label="Describe el perfil que buscas"),
    outputs=gr.Markdown(label="Resultado del sistema"),
    title="CV Matcher AI",
    description="Busca el mejor CV basado en tu query usando embeddings, FAISS y un LLM local."
)

if __name__ == "__main__":
    ui.launch()
