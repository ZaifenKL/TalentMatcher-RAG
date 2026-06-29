import os
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
##------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Modules that need __init__.py
folders = [
    "Build_Vector_DBase",
    "Chunking",
    "Embeddings",
    "Feed_and_processing",
    "LLM_Engine",
    "UI",
    "Job_Matching",
    "Report_Builder",
    "Test"
]
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def create_init(folders):
    for folder in folders:
        folder_path = os.path.join(PROJECT_ROOT, folder)
        init_path = os.path.join(folder_path, "__init__.py")

        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder does not exist: {folder_path}")
            continue

        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                pass
            print(f"[OK] Created __init__.py at: {init_path}")
        else:
            print(f"[INFO] Already exists: {init_path}")

##----DEBUG/TEST------------------------------------------------------------------
create_init(folders)
