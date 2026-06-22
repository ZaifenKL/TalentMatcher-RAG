#Step 1 : Load the pdf knowledge we need to feed our RAG
#Import libraries
import os
import json
from collections import defaultdict
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
##------------------------------------------------------------------------
base_path = r"C:\AI Stuff\CV_Matching_AI\Data\CV\Base_Profile"
cvs_path = r"C:\AI Stuff\CV_Matching_AI\Data\CV_Test"
json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Raw_Json"
##--------------------Variables------------------------------------------
##------------------------------------------------------------------------
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def ensure_layer_exists(output_path, layer_name):
    #Creates the folder layer fo json if it does not exist
    layer_path = os.path.join(output_path, layer_name)
    os.makedirs(layer_path, exist_ok=True)
    return layer_path

def get_json_paths(base_path):
    #Returns a set of the existing json paths
    json_paths = set()
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(".json"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_path)
                json_paths.add(rel_path.replace(".json", ".pdf"))
    return json_paths

def get_pdf_paths(base_path):
    #Return a list of the existing pdf files
    pdf_paths = set()
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_path)
                pdf_paths.add(rel_path)
    return pdf_paths

def pdf_missing_convertion(base_path, json_path,show_results=False):
    missing_by_layer = defaultdict(list)
    total_missing = 0
    pdf_set = get_pdf_paths(base_path)
    json_set = get_json_paths(json_path)
    if len(pdf_set) != len(json_set):
        print(f"Total PDFs: {len(pdf_set)}")
        print(f"Total JSONs: {len(json_set)}")
        for pdf_path in pdf_set:
            if pdf_path in json_set:
                continue  # If it exists in both sets jump to the next iteration
            # Obtain layer
            parts = pdf_path.split(os.sep)
            layer = parts[0] if len(parts) > 1 else "NO_LAYER"
            # Only the name of the file
            file_name = os.path.basename(pdf_path)
            missing_by_layer[layer].append(file_name)
        print(f"Layers with missing: {len(missing_by_layer)}")
    if show_results:
        for layer, files in missing_by_layer.items():
            print(layer, "->", len(files))
            total_missing = sum(len(files) for files in missing_by_layer.values())
        print("TOTAL missing:", total_missing)
    return dict(missing_by_layer)

def ini_converter():
    #This function initializes the docling converter
    pipeline_options = PdfPipelineOptions()
    # Explicitly swap the backend engine away from docling-parse
    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=PyPdfiumDocumentBackend # Swapping backends prevents the C++ memory leak
        )
    }
    #Initialize docling converter from pdf to text
    converter = DocumentConverter(format_options=format_options)
    return converter

def convert_pdf_to_json(base_path, missing_by_layer, show_results=False):
    #Function reads pdf files and extracts content into JSON with markdown
    #Initialize docling converter
    converter = ini_converter()
    for layer, pdf_list in missing_by_layer.items():
        ensure_layer_exists(json_path, layer)
        if show_results:
            print(f"\n=== Processing Layer: {layer} ===")
        for pdf in pdf_list:
            pdf_path = os.path.join(base_path, layer, pdf)
            try:
                if show_results:
                    print("Processing:", pdf_path)
                    print("Name:", pdf)
                # Convert PDF using Docling
                result = converter.convert(pdf_path)
                doc = result.document

                # --- Accumulators ---
                text_parts = []
                table_parts = []
                table_count = 0

                # --- Extract TEXT ---
                for t in doc.texts:
                    if hasattr(t, "text") and t.text:
                        text_parts.append(t.text)
                # --- Extract TABLES ---
                for tbl in doc.tables:
                    try:
                        table_parts.append(tbl.export_to_markdown(doc))
                    except:
                        pass

                # --- Concat blocks ---
                text_block = "\n\n".join(text_parts).strip()
                table_block = "\n\n".join(table_parts).strip()

                # --- Build JSON -------------------
                json_data = {
                    "name": pdf,
                    "source_path": pdf_path,

                    "text_block": {
                        "content": text_block,
                        "char_count": len(text_block)
                    },

                    "table_block": {
                        "content": table_block,
                        "table_count": table_count,
                        "char_count": len(table_block)
                    }
                }

                # --- Save JSON ---
                json_filename = os.path.splitext(pdf)[0] + ".json"
                json_output = os.path.join(json_path, layer, json_filename)

                with open(json_output, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"====JSON saved: {json_filename}====")
            except Exception as e:
                print(f"Error en {pdf}: {e}")

##----DEBUG/TEST----------------------------------------------------------------------
# Step 1) Call pdf_missing convertion and convert pdf to text in JSON markdown format
#json_paths = get_json_paths(json_path)
#pdf_paths = get_pdf_paths(base_path)
#print(f"\n{len(pdf_paths)} PDFs found")
#print(f"\n{len(json_paths)} JSONs found")
#Main functions
missing_by_layer = pdf_missing_convertion(base_path,json_path,show_results=False)
convert_pdf_to_json(base_path,missing_by_layer, show_results=True)