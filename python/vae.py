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
        #pour entree doit avoir la taille du vocabulaire car le couche linear prends en entree les vecteurs de caractere individuellement 
        self.encoder = torch.nn.Sequential(torch.nn.Linear(29, 32),  # Assuming input length is 57
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
                                             torch.nn.Linear(32, 29),
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
        #print("shape x_recon forward before view ",x_recon.shape)
        #x_recon = x_recon.view(-1, 64, 29)
        #print("shape x_recon forward ",x_recon.shape)
        return x_recon, mu, logvar
    
    def vae_loss(self, x_recon, x, mu, logvar):
        batch_size, seq_len, vocab_size = x_recon.size()

        # Aplatir pour correspondre à l'input de F.cross_entropy
        x_recon = x_recon.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
        x = x.view(-1)                          # (batch_size * seq_len)

        # Calcul de la reconstruction loss
        recon_loss = torch.nn.functional.cross_entropy(x_recon, x, reduction='sum')

        # Calcul de la divergence KL
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_div
    

# Créer le vocabulaire
def build_vocab(smiles_list):
    charset = set() #set car enleve les doublons
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

def one_hot_encoded_smiles(smiles,max_len,char_to_idx):
    """
    Encode a SMILES string into a one-hot encoded vector.
    
    Parameters:
    smiles (str): The SMILES string to encode.
    
    Returns:
    np.ndarray: One-hot encoded vector of the SMILES string.
    """
    one_hot = np.zeros((max_len, len(char_to_idx)), dtype=np.float32)
    
    for i, char in enumerate(smiles):
        if i < max_len:
            one_hot[i, char] = 1.0
            
    return one_hot


if __name__=="__main__":

    # Ajoute le dossier parent (dossier_python) au path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import data_manipulation as dm


    #refaire les chemins plus tard pour appel dans rep parent
    parent_path = os.path.abspath("./data/moses/train.csv")  # Aller au dossier parent

    
    print(parent_path)

    # Check if CUDA is available and print the GPU name
    #print(torch.cuda.is_available())      # True si CUDA est détecté
    #print(torch.__version__)
    #print(torch.version.cuda)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="cpu"
    
    #print(torch.cuda.get_device_name(0))  # Affiche le nom de ton GPU


    moses_train = dm.load_data(parent_path)
   
    SMILES_list=moses_train["SMILES"].to_list()

    char_to_idx, idx_to_char = build_vocab(SMILES_list)
    


    print("Vocabulaire construit avec", len(char_to_idx), "caractères.")
    print("max idx:", max(char_to_idx.values()))
    first_encoded_smile= encode_smiles(SMILES_list[0],64)
    print("Exemple de SMILES encodé:",first_encoded_smile)
    one_hot_first_smile=one_hot_encoded_smiles(first_encoded_smile,64,char_to_idx)
    print("test one hot ", one_hot_first_smile)
    print("shape one hot ", one_hot_first_smile.shape)

    #tensor_one_hot_first_smile=torch.tensor(one_hot_first_smile)
    #ajout de la dimension pour le format du batch
    tensor_one_hot_first_smile=torch.tensor(one_hot_first_smile).unsqueeze(0)
    print("shape one hot tensor ",tensor_one_hot_first_smile.shape)

    model=VAE()
    print(model)

    output=model(tensor_one_hot_first_smile)
    #print("sortie du model \n",output)
    print("shape de l output: ",output[0].shape)

    #exit()
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
    encode_smiles_l=list(map( lambda x:one_hot_encoded_smiles(x,64,char_to_idx),encode_smiles_l))
    #print(encode_smiles_l[0:5])
    l_encoded_smiles = len(encode_smiles_l[0])
    for k in range(len(encode_smiles_l)):
        break
        if l_encoded_smiles != len(encode_smiles_l[k]):
            print("Error at index", k)
            print("Length of encoded SMILES:", len(encode_smiles_l[k]))
            print("Expected length:", l_encoded_smiles)
        
    l_encoded_smiles=np.array(l_encoded_smiles)
    X_train = torch.tensor(encode_smiles_l, dtype=torch.float,device="cpu")

    # Test de l'encodeur
    vae = VAE()
    vae.to(device)
    
    print("shape du train ",X_train[0].shape)
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