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
##--------------------Variables------------------------------------------
##------------------------------------------------------------------------
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
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

def convert_pdf_to_json(cvs_path, json_path, show_results=False):
    #Function reads pdf files and extracts content into JSON with markdown
    #Initialize docling converter
    converter = ini_converter()
    for root, dirs, files in os.walk(cvs_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                ext = os.path.splitext(file)[1].upper()  # ".pdf"
                file_type = ext.replace(".", "")  # "pdf"
                print("===Processing:", file)
                try:
                    #Convert PDF using Docling
                    file_path = os.path.join(root, file)
                    result = converter.convert(file_path)
                    doc = result.document

                    # --- Accumulators ---
                    text_parts = []
                    # --- Extract TEXT ---
                    for t in doc.texts:
                        if hasattr(t, "text") and t.text:
                            text_parts.append(t.text)

                    # --- Ensure folder exists ---
                    output_dir = os.path.join(json_path, file_type)
                    os.makedirs(output_dir, exist_ok=True)

                    # --- Build JSON -------------------
                    json_data = {
                        "name": file,
                        "source_path": file_path,
                        "source_format": file_type,

                        "text": {
                            "content": text_parts,
                            "element_count": len(text_parts)
                        }
                    }

                    # --- Save JSON ---
                    json_filename = os.path.splitext(file)[0] + ".json"
                    json_output = os.path.join(json_path,file_type,json_filename)
                    with open(json_output, "w", encoding="utf-8") as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)
                    print(f"====JSON saved: {json_filename}====")

                except Exception as e:
                 print(f"Error en {file}: {e}")

##----DEBUG/TEST----------------------------------------------------------------------
convert_pdf_to_json(cvs_path,json_path, show_results=True)