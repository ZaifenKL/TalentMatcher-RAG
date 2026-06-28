import configparser
import os
##-------Define functions to optimize the pdf extraction data flow--------
##------------------------------------------------------------------------

def load_config():
    config = configparser.ConfigParser()

    #Absolute path for config.ini
    base_path = os.path.dirname(os.path.dirname(__file__))
    ini_path = os.path.join(base_path, "config.ini")

    config.read(ini_path, encoding="utf-8")
    return config

config = load_config()
