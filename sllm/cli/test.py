from transformers import BitsAndBytesConfig
import json

config = BitsAndBytesConfig(load_in_4bit=True)
config.to_json_file("quantization_config.json")
