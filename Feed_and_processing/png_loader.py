#Step 1 : Load the pdf knowledge we need to feed our RAG
#Import libraries
import os
import json
from PIL import Image
import pytesseract
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
                    img = Image.open(file_path)
                    text = pytesseract.image_to_string(img)
                    # --- Build JSON ----------------------
                    json_data = {
                        "name": file,
                        "source_path": file_path,
                        "source_format": file_type,
                        "text": {
                            "content": text,
                            "char_count": len(text)
                        }
                    }

                    # --- Ensure folder exists ---
                    output_dir = os.path.join(json_path,file_type)
                    os.makedirs(output_dir, exist_ok=True)

                    # --- Save JSON ---
                    json_filename = os.path.splitext(file)[0] + ".json"
                    json_output = os.path.join(output_dir, json_filename)

                    with open(json_output, "w", encoding="utf-8") as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)

                    print(f"====JSON saved: {json_filename}====")

                except Exception as e:
                    print(f"Error en {file}: {e}")


##----DEBUG/TEST----------------------------------------------------------------------
convert_png_to_json(cvs_path, json_path)