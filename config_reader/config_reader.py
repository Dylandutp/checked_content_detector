import configparser
import os

class ConfigReader():
    def __init__(self):
        self.path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..")), 
                                "config", "config.ini")
        self.cf = configparser.ConfigParser()
        self.cf.read(self.path)

    def __write_config(self):
        with open(self.path, "w+") as f:
            self.cf.write(f)

    def get_root_path(self):
        return self.cf.get("path", "root_path")
