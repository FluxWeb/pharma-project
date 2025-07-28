import torch
import pandas as pd
import numpy as np

import sys
import os




# class from chem VAE article
"""
The structure of the VAE deep network was as follows: For the autoencoder used for the 
ZINC data set, the encoder used three 1D convolutional layers of filter sizes 9, 9, 10 
and 9, 9, 11 convolution kernels, respectively, followed by one fully connected layer of width 196. 
The decoder fed into three  layers of gated recurrent unit (GRU) networks53 with hidden dimension of 488.
For the model used for the QM9 data set, the encoder used three 1D convolutional layers of filter
sizes 2, 2, 1 and 5, 5, 4 convolution kernels, respectively, followed by one fully connected layer 
of width 156. The three recurrent neural network layers each had a hidden dimension of 500 neurons. 
The last layer of the RNN decoder defines a probability distribution over all possible characters 
at each position in the SMILES string.
"""

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Conv1d(in_channels=9,out_channels=9, kernel_size=9),
                                           torch.nn.ReLU(),
                                           torch.nn.Conv1d(in_channels=9, out_channels=10, kernel_size=9),
                                           torch.nn.ReLU(),
                                           torch.nn.Conv1d(in_channels=10, out_channels=11, kernel_size=10),
                                           torch.nn.ReLU(),
                                           torch.nn.Flatten())
        self.decoder = torch.nn.Sequential()

    def encode(self, x):
        params = self.encoder(x)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def tret_encoder(self, x):
        """
        Fonction pour tester l'encodeur
        """
        print(x.shape)

        return self.encoder(x)
    

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


    #refaire les chemins plus tard pour appel dans rep parent
    parent_path = os.path.abspath("../../data/moses/train.csv")  # Aller au dossier parent

    
    print(parent_path)

    # Check if CUDA is available and print the GPU name
    print(torch.cuda.is_available())      # True si CUDA est détecté
    
    #print(torch.cuda.get_device_name(0))  # Affiche le nom de ton GPU


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

    # Test de l'encodeur
    vae = VAE()
    res=vae.tret_encoder(torch.randn(1, 9, 57))  # Exemple de test avec un batch de taille 1, 9 canaux, longueur 57
    print(res)