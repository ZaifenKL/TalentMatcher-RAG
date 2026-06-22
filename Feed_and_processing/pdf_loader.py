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
cvs_path = r"C:\AI Stuff\CV_Matching_AI\Data\CV\CV_Test"
json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Raw_Json"
json_pdf_path = r"/Data/Raw_Json/PDF"
json_png_path = r"/Data/Raw_Json/PNG"
##--------------------Variables------------------------------------------
##------------------------------------------------------------------------
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def get_cvs_paths(cvs_path):
    #Returns a list of the existing pdf files and png files
    pdf_paths = set()
    png_paths = set()
    for root, dirs, files in os.walk(cvs_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_path)
                pdf_paths.add(rel_path)
            if file.lower().endswith(".png"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_path)
                png_paths.add(rel_path)
    return pdf_paths,png_paths

def get_json_paths(json_path):
    #Returns a set of the existing json paths
    json_paths = set()
    for root, dirs, files in os.walk(json_path):
        for file in files:
            if file.lower().endswith(".json"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_path)
                json_paths.add(rel_path)
    return json_paths

def missing_convertion(cvs_path, json_pdf_path,json_png_path,show_results=False):
    #Returns the list of non-converted CVs so if something new is added we will not need to
    # re-process again the entire list of CVs
    total_missing = 0
    missing_cvs = set()
    pdf_set, png_set = get_cvs_paths(cvs_path)
    union_cv_set = pdf_set.union(png_set)
    json_pdf_set = get_json_paths(json_pdf_path)
    json_png_set = get_json_paths(json_png_path)
    union_json_set = json_pdf_set.union(json_png_set)
    if len(union_cv_set) != len(union_json_set):
        if show_results:
            print(f"Total CVs: {len(union_cv_set)}")
            print(f"Total JSONs: {len(union_json_set)}")
            print(f"json pdf set: {json_pdf_set}")
            print(f"json png set: {json_png_path}")
            print(f"union cv set: {union_cv_set}")





def ensure_layer_exists(output_path, layer_name):
    #Creates the folder layer fo json if it does not exist
    layer_path = os.path.join(output_path, layer_name)
    os.makedirs(layer_path, exist_ok=True)
    return layer_path

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

def convert_to_json(cvs_path, show_results=False):
    #Function reads pdf files and extracts content into JSON with markdown
    #Initialize docling converter
    converter = ini_converter()
    for root, dirs, files in os.walk(cvs_path):
        for file in files:
            ext = os.path.splitext(file)[1].upper()  # ".pdf" or ".png"
            file_type = ext.replace(".", "")  # "pdf" or "png"
            print("===Processing:", file)
            if file.lower().endswith(".pdf"):
                #Convert PDF using Docling
                result = converter.convert(root)
                doc = result.document

                # --- Accumulators ---
                text_parts = []
                # --- Extract TEXT ---
                for t in doc.texts:
                    if hasattr(t, "text") and t.text:
                        text_parts.append(t.text)

                # --- Build JSON -------------------
                json_data = {
                    "name": file,
                    "source_path": root,
                    "source_format": file_type,

                    "text": {
                        "content": text_parts,
                        "char_count": len(text_parts)
                    }
                }

                # --- Save JSON ---
                json_filename = os.path.splitext(file)[0] + ".json"
                json_output = os.path.join(json_path,ext,json_filename)
                with open(json_output, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"====JSON saved: {json_filename}====")
            except Exception as e:
                print(f"Error en {file}: {e}")

##----DEBUG/TEST----------------------------------------------------------------------
# Step 1) Call pdf_missing convertion and convert pdf to text in JSON markdown format
#json_paths = get_json_paths(json_path)
#pdf_paths , png_paths = get_cvs_paths(cvs_path)
#print(f"pdf_paths: {pdf_paths}")
#print(f"png_paths: {png_paths}")
#json_paths = get_json_paths(json_path)
#print(f"json_paths: {json_paths}")
#missing_convertion(cvs_path, json_pdf_path,json_png_path,show_results=True)
convert_to_json(cvs_path, show_results=True)