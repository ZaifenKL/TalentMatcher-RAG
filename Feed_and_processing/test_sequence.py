from Feed_and_processing import sanitize_filenames,convert_pdf_to_json, convert_png_to_json, clean_json
cvs_path = r"C:\AI Stuff\CV_Matching_AI\Data\CV\CV_Test"
json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Raw_Json"
raw_json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Raw_Json"
clean_json_path = r"C:\AI Stuff\CV_Matching_AI\Data\Clean_Json"
HEADERS = ["Formación", "Educación","Experiencia Profesional","Habilidades Técnicas","Habilidades Blandas","Certificaciones",
           "Idiomas", "Proyectos Relevantes", "Formación académica"]


def main():
    print("\n=== Sanitizing Filenames in CV Folder ===")
    sanitize_filenames(cvs_path, show_result=True)
    print("\n=== Sanitization Complete ===")
    print("\n=== Extracting PDF Content into Raw JSON ===")
    convert_pdf_to_json(cvs_path, json_path, show_results=True)
    print("\n=== PDF Extraction Complete ===")
    print("\n=== Extracting OCR from PNG/JPG Files ===")
    convert_png_to_json(cvs_path, json_path)
    print("\n=== OCR Extraction Complete ===")
    print("\n=== Cleaning Raw JSON Files ===")
    clean_json(raw_json_path, clean_json_path, show_results=True)
    print("\n=== Cleaning Complete ===")

if __name__ == "__main__":
    main()