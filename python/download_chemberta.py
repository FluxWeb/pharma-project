import transformers
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import os

#pas utile pour l instant sans doute à retirer
#pour l instant on utilise le VAE pour generer des smiles


if __name__=="__main__":
    print("Main downloading model chemBerta")

    #attention à l'arboresence du projet pour cette partie
    save_dir="./python/model/chemberta"
    print("Saving model in: ", os.path.abspath(save_dir))
    #print(os.path.isdir(save_dir))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Created directory for model:", save_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", cache_dir=save_dir,local_files_only=True)
        model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", cache_dir=save_dir,local_files_only=True)
    except Exception as e:
        print("model not already downloaded")
        print("Downloading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1",
                                                cache_dir=save_dir,
                                                trust_remote_code=True,    
                                                local_files_only=False,
                                                torch_dtype=torch.float32,
                                                use_safetensors=True,  # ✅ IMPORTANT
                                                )
        model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1",
                                                cache_dir=save_dir,
                                                trust_remote_code=True,    
                                                local_files_only=False,
                                                torch_dtype=torch.float32,
                                                use_safetensors=True,  # ✅ IMPORTANT
                                                )