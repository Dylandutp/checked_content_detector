import os
import configparser
from main import main

if __name__ == '__main__':
    # Create config file
    root_path = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(root_path, "config", "config.ini")
    config = configparser.ConfigParser()
    config.read(config_path)
    setting = config.items("setting")
    config['setting'] = {key: value for key, value in setting}
    config['path'] = {'ROOT_PATH': root_path}
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    print(f"Create config file at '{config_path}'")

    main()