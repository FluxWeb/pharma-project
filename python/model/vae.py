import torch
import pandas as pd
import numpy as np

import sys
import os


smiles_list = ['CCO', 'CC(=O)O', 'CCN(CC)CC', 'c1ccccc1']

# Créer le vocabulaire
def build_vocab(smiles_list):
    charset = set()
    for smiles in smiles_list:
        for char in smiles:
            charset.add(char)
    charset = sorted(list(charset)) + ['<pad>', '<start>', '<end>']
    char_to_idx = {c: i for i, c in enumerate(charset)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char

char_to_idx, idx_to_char = build_vocab(smiles_list)
vocab_size = len(char_to_idx)

# Encode SMILES
def encode_smiles(smiles, max_len):
    tokens = ['<start>'] + list(smiles) + ['<end>']
    tokens += ['<pad>'] * (max_len - len(tokens))
    return [char_to_idx[c] for c in tokens]

# Décode
def decode_smiles(indices):
    return ''.join([idx_to_char[i] for i in indices if idx_to_char[i] not in ['<start>', '<pad>', '<end>']])



if __name__=="__main__":

    # Ajoute le dossier parent (dossier_python) au path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import data_manipulation as dm



    parent_path = os.path.abspath("../data/moses/train.csv")  # Aller au dossier parent
    print(parent_path)

    # Check if CUDA is available and print the GPU name
    print(torch.cuda.is_available())      # True si CUDA est détecté
    print(torch.cuda.get_device_name(0))  # Affiche le nom de ton GPU


    moses = dm.load_data(parent_path)
   
    SMILES_list=moses["SMILES"].to_list()

    char_to_idx, idx_to_char = build_vocab(SMILES_list)
    
    """
    #test pour verifier isomophisme du decodeur
    test_encode=moses["SMILES"].apply(lambda x: encode_smiles(x, max_len=57)).to_list()
    print(test_encode[0:5])
    test_decode=list(map(decode_smiles,test_encode))
    print(test_decode[0:5])
    for i in range(len(moses["SMILES"])):
        
        if test_decode[i] != moses["SMILES"][i]:
            print("Error at index", i)
            print("Original SMILES:", moses["SMILES"][i])
            print("Decoded SMILES:", decode_smiles(test_encode[i]))
            
        elif i==len(moses["SMILES"])-1 and test_decode[i] == moses["SMILES"][i]:
            print("All SMILES decoded correctly.")
            break
    """
