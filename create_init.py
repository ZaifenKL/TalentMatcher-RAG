import os
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
##------------------------------------------------------------------------
# Modules that need __init__.py
folders = [
    "Build_Vector_DBase",
    "Chunking",
    "Embeddings",
    "Feed_and_processing",
    "LLM_Engine",
    "UI"
]
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def create_init(folders):
    for folder in folders:
        init_path = os.path.join(folder, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                pass
            print(f"Created init at: {init_path}")
        else:
            print(f"Module already has: {init_path}")

##----DEBUG/TEST------------------------------------------------------------------
create_init(folders)
