import os
import re
import json
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
##------------------------------------------------------------------------
raw_json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Raw_Json"
clean_json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Clean_Json"
HEADERS = ["Formación", "Educación","Experiencia Profesional","Habilidades Técnicas","Habilidades Blandas","Certificaciones",
           "Idiomas", "Proyectos Relevantes"]
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def clean_extra_spaces(text):
    #Remove extra spaces in text
    text = re.sub(r"\s+", " ", text)
    # 2 Remove spaces at the beginign and at the end
    text + re.sub(r'^\s+|\s+$', '', text)

    return text

def fix_ocr_zero_to_o(text) :
    #Replace 0 for o if its between letters
    # Case 1: letter + 0 + letter should replace zero for letter o
    text = re.sub(r'(?i)([A-Za-zÁÉÍÓÚÜÑáéíóúüñ])0([A-Za-zÁÉÍÓÚÜÑáéíóúüñ])',
                  r"\1o\2", text)
    # Case 2: space + 0 + space → replace for " o "
    text = re.sub(r'\s0\s', ' o ', text)
    #Case 3 : word ending in "0" should be replaced by "o"
    text = re.sub(
        r'(?i)\b([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)0\b',
        r'\1o',text)

    return text

def add_colons_to_headers(text,HEADERS):
    #Add " : " to headers will help the model understand groups
    HEADERS = [s.strip().lower() for s in HEADERS] #Normalize list of headers
    stripped_text = text.strip().lower() #Remove spaces and lower case to compare in given line/text
    if stripped_text in HEADERS:
        text = text + " : "

    return text

def join_clean_lines(lines):
    # 1. Quitar líneas vacías o con solo espacios
    lines = [l.strip() for l in lines if l.strip()]

    # 2. Unir con salto de línea
    text = "\n".join(lines)

    # 3. Normalizar espacios finales
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    return text

def clean_json(raw_json_path, clean_json_path):
    #This function creates the clean json we will use for chunks
    for root, dirs, files in os.walk(raw_json_path):
        for file in files:
            if file.lower().endswith(".json"):
                print("===Processing:", file)
                try:
                    #Complete path of json input file
                    in_path = os.path.join(root, file)
                    with open(in_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # Load json content into a variable
                        raw_lines = data["text"]["content"]

                        clean_lines = []
                        for line in raw_lines:
                            line = clean_extra_spaces(line)
                            line = fix_ocr_zero_to_o(line)
                            line = add_colons_to_headers(line, HEADERS)
                            clean_lines.append(line)
                            clean_content = join_clean_lines(clean_lines)

                        # --- Ensure folder exists ---
                        output_dir = os.path.join(clean_json_path, data["source_format"])
                        os.makedirs(output_dir, exist_ok=True)

                        # --- Build Clean JSON -------------------
                        json_data = {
                            "name": data["name"],
                            "source_path": data["source_path"],
                            "source_format": data["source_format"],

                            "text": {
                                "content": clean_content,
                                "element_count": len(clean_content),
                                 }
                             }

                        # --- Save JSON ---------
                        json_filename = os.path.splitext(file)[0] + ".json"
                        json_output = os.path.join(clean_json_path, data["source_format"], json_filename)
                        with open(json_output, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)
                        print(f"====JSON saved: {json_filename}====")
                except Exception as e:
                    print(f"Error proccesing {file}: {e}")

##----DEBUG/TEST------------------------------------------------------------------
#Text = ["Backend        Engineer     (Intermedi0)",
#     "   Resumen    Profesi0nal",
#     "Profesional de   Backend Engineer con 3-6 años de experiencia. Perfil orientado a resultados, "
#     "análisis de neg0cio, desarrollo de software y trabajo colab0rativo. Inspirado en el estilo de CV profesional "
#     "observado en los ejemplos proporcionados.  ",
#     "Formación",
#     "Licenciatura en Ingeniería en   Sistemas, Computación,    Matemáticas Aplicadas 0 afín. "]

clean_json(raw_json_path, clean_json_path)
