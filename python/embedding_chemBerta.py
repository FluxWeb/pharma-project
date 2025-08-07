import transformers
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

#print(torch.__version__)



moses_df =pd.read_csv("./data/moses/train.csv")


model_path = "./model/chemberta/seyonec/ChemBERTa-zinc-base-v1"

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1",
                                          local_files_only=True)
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1",
                                  local_files_only=True, 
                                  use_safetensors=True)

first_smile = moses_df.iloc[0]['SMILES']

#print("SMILES: ", first_smile)
#print("Tokenized:", tokenizer.tokenize(first_smile))
#print("Model output:", model(**tokenizer(first_smile, return_tensors='pt')))

first_smile_tokenized = tokenizer(first_smile, return_tensors='pt')
model_output = model(**first_smile_tokenized)

print("first smile length:", len(first_smile))
print("Model output shape:", model_output.last_hidden_state.shape)