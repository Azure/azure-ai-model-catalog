import os
 
def get_huggingface_token():
    return os.environ.get('HF_token', '')
