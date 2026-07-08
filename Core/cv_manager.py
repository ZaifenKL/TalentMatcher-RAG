import os
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------

##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------

def detect_new_cvs(cv_path, embed_json_path):
    # Available CV (PDF/PNG)
    cv_files = [
        f for f in os.listdir(cv_path)
        if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg"))
    ]

    # Processed CVs (embedding json)
    processed = {
        f.replace(".json", "")
        for f in os.listdir(embed_json_path)
        if f.endswith(".json")
    }

    #New CVs.. files without embeddings
    new_cvs = []

    for cv in cv_files:
        name = os.path.splitext(cv)[0]
        if name not in processed:
            new_cvs.append(cv)

    return new_cvs
