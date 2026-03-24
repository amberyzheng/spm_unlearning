import os
import yaml 
import importlib

def load_config(file_path: str) -> dict:

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)
    except Exception as e:
        raise ValueError(f"Failed to read or parse config file {file_path}: {e}")
    
    if config_data is None:
        raise ValueError(f"Configuration file {file_path} is empty or not properly formatted.")
    
    return config_data



def instantiate_from_config(config):
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)