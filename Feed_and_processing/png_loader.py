#Step 1 : Load the pdf knowledge we need to feed our RAG
#Import libraries
import os
import json
import easyocr
from Core.config_loader import config
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
##------------------------------------------------------------------------
cvs_path = r"C:\AI Stuff\CV_Matching_AI\Data\CV\CV_Test"
json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Raw_Json"
##--------------------Variables------------------------------------------
##------------------------------------------------------------------------
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def convert_png_to_json(cvs_path, json_path):
    reader = easyocr.Reader(['en', 'es'])  #languages
    #Converts png files into JSON
    for root, dirs, files in os.walk(cvs_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                ext = os.path.splitext(file)[1].upper()  # ".jpg" or ".png"
                file_type = ext.replace(".", "")  # "jpg" or "png"
                try:
                    print("===Processing :", file)
                    file_path = os.path.join(root, file)

                    # --- OCR extraction ---
                    # detail=0 returns a list of strings (perfect for RAW JSON)
                    text_parts = reader.readtext(file_path, detail=0)

                    # --- Build JSON (RAW style) ---
                    json_data = {
                        "name": file,
                        "source_path": file_path,
                        "source_format": file_type,
                        "text": {
                            "content": text_parts,  # LISTA DE STRINGS
                            "char_count": len(" ".join(text_parts))  # TOTAL DE CARACTERES
                        }
                    }

                    # Save JSON
                    output_dir = os.path.join(json_path, file_type)
                    os.makedirs(output_dir, exist_ok=True)

                    json_filename = os.path.splitext(file)[0] + ".json"
                    json_output = os.path.join(output_dir, json_filename)

                    with open(json_output, "w", encoding="utf-8") as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)

                    print(f"====JSON saved: {json_filename}====")

                except Exception as e:
                    print(f"Error en {file}: {e}")

##----DEBUG/TEST----------------------------------------------------------------------
def main():
    print("\n=== Extracting OCR from PNG/JPG Files ===")
    convert_png_to_json(cvs_path, json_path)
    print("\n=== OCR Extraction Complete ===")


if __name__ == "__main__":
    main()