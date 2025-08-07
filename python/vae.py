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
        super().__init__()
        """
        self.encoder = torch.nn.Sequential(torch.nn.Conv1d(in_channels=1, out_channels=9, kernel_size=1),
                                           torch.nn.MaxPool1d(3),
                                           torch.nn.Conv1d(in_channels=9,out_channels=9, kernel_size=9),
                                           torch.nn.MaxPool1d(3),
                                           torch.nn.Conv1d(in_channels=9, out_channels=10, kernel_size=9),
                                           torch.nn.MaxPool1d(3),
                                           torch.nn.Conv1d(in_channels=10, out_channels=11, kernel_size=10),
                                           torch.nn.MaxPool1d(3),
                                           torch.nn.Flatten()
                                           )
        """

        #simple VAE for try and advancing
        self.encoder = torch.nn.Sequential(torch.nn.Linear(64, 32),  # Assuming input length is 57
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(32, 16),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(16, 8),
                                             torch.nn.ReLU(),
                                             
                                             )

        self.mu = torch.nn.Linear(8, 2)

        self.logvar= torch.nn.Linear(8, 2)

        self.decoder = torch.nn.Sequential(torch.nn.Linear(2, 8),  # Assuming input length is 57
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(8, 16),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(16, 32),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(32, 64),
                                             torch.nn.ReLU()
                                             )

    
 
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # même taille que std, valeurs aléatoires ~ N(0, 1)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def vae_loss(self, x_recon, x, mu, logvar):
        recon_loss = torch.nn.functional.cross_entropy(x_recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div
    

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
    #print(torch.cuda.is_available())      # True si CUDA est détecté
    #print(torch.__version__)
    #print(torch.version.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(torch.cuda.get_device_name(0))  # Affiche le nom de ton GPU


    moses_train = dm.load_data(parent_path)
   
    SMILES_list=moses_train["SMILES"].to_list()

    char_to_idx, idx_to_char = build_vocab(SMILES_list)
    
    """
    #test pour verifier isomophisme du decodeur
    test_encode=moses_train["SMILES"].apply(lambda x: encode_smiles(x, max_len=57)).to_list()
    print(test_encode[0:5])
    test_decode=list(map(decode_smiles,test_encode))
    print(test_decode[0:5])
    for i in range(len(moses_train["SMILES"])):
        
        if test_decode[i] != moses_train["SMILES"][i]:
            print("Error at index", i)
            print("Original SMILES:", moses_train["SMILES"][i])
            print("Decoded SMILES:", decode_smiles(test_encode[i]))
            
        elif i==len(moses_train["SMILES"])-1 and test_decode[i] == moses_train["SMILES"][i]:
            print("All SMILES decoded correctly.")
            break
    """
    max_len = moses_train["SMILES"].apply(lambda x: len(x)).max() + 2  # +2 for <start> and <end>

    encode_smiles_l= moses_train["SMILES"].apply(lambda x: encode_smiles(x, 64)).to_list()
    #print(encode_smiles_l[0:5])
    l_encoded_smiles = len(encode_smiles_l[0])
    for k in range(len(encode_smiles_l)):
        break
        if l_encoded_smiles != len(encode_smiles_l[k]):
            print("Error at index", k)
            print("Length of encoded SMILES:", len(encode_smiles_l[k]))
            print("Expected length:", l_encoded_smiles)
        
    X_train = torch.tensor(encode_smiles_l, dtype=torch.float).to(device)

    # Test de l'encodeur
    vae = VAE()
    vae.to(device)
    
    test=vae(X_train[0])  # Ajouter des dimensions pour correspondre à l'entrée attendue

    print(test)

    optimizer = torch.optim.Adam(params=vae.parameters(), lr=0.001)
    
    epochs=100

    for epoch in range(epochs):
    
        vae.train()

        pred=vae(X_train)

        loss=vae.vae_loss(pred[0],X_train, pred[1], pred[2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if epoch%100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
        """
        with torch.inference_mode():
            vae.eval()
            pred=vae(X_train)
            loss=vae.vae_loss(pred[0],X_train, pred[1], pred[2])
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
        """

    test_new_vec=torch.randn(1, 2).to(device)  # Générer un vecteur aléatoire de dimension 2
    decoded_smiles = vae.decode(test_new_vec)

    print("Decoded SMILES from random vector:", decoded_smiles)