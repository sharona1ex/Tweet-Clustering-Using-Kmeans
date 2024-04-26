import json

model_config_path = "configs/model_config.json"
data_config_path = "configs/data_config.json"


def load_config(path):
    with open(path, "r") as file:
        content = json.load(file)
    return content


# use these variables from here
MODEL = load_config(model_config_path)
DATA = load_config(data_config_path)