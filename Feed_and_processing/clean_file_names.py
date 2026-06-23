import os
import unicodedata
##------------------------------------------------------------------------
##------------------Constant Values---------------------------------------
##------------------------------------------------------------------------
cvs_path = r"C:\AI Stuff\CV_Matching_AI\Data\CV\CV_Test"
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------
def sanitize_filenames(base_path,show_result=False):
    #Normalize file names
    renamed_count = 0
    for root, dirs, files in os.walk(base_path):
        for file in files:
            original_path = os.path.join(root, file)
            # Normalize ASCII remove accent
            normalized = unicodedata.normalize('NFKD', file).encode('ascii', 'ignore').decode('ascii')

            #Repalce spaces with " _ "
            normalized = normalized.replace(" ", "_")

            #Id file name changed then rename
            if normalized != file:
                new_path = os.path.join(root, normalized)
                os.rename(original_path, new_path)
                renamed_count += 1
                if show_result:
                    print(f"Renamed: {file} → {normalized}")
    if show_result:
        print(f"\nTotal renamed files: {renamed_count}")

##----DEBUG/TEST------------------------------------------------------------------
sanitize_filenames(cvs_path,True)