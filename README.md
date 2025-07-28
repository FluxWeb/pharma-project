# molecularb generation project


necessite python 3.12 ou moins pour installer torch avec CUDA

j utilise la version 3.11

compiler avec la version python 
py -3.11 mon_script.py

mettre à jour pip

### creation d environnement 

py -3.11 -m venv mol_gen

### activer environnement

In cmd.exe pas par default dans visual studio 
mol_gen\Scripts\activate.bat

In PowerShell (attention il faut definir une strategie d execution python)
mol_gen\Scripts\Activate.ps1

### confirmer activation

where python

### voir PATH

cmd
echo %PATH%

powershell
\$Env:Path 

### deactiver
deactivate

### supprimer environnement
rm -r mol_gen

windows: rmdir /S /Q mol_gen


pour recuperer les donnees directement dans pip dans le requirements.txt il faut run la commande
    pip freeze > requirements.txt

## datasets

MOSES : https://github.com/molecularsets/moses


## Deep Learning Model for generation

VAE
Normalizing Flows
GAN
Diffusion Model

## embedding des smiles

representation numerique fait intuitivement en listant tout les carateres il doit exister de meilleur representation a chercher

# TO DO::

priorite au VAE pour tester et implementer une representation des molecules
voir si API pour verifier la validite des molecules generes

prendre une journee et recfhercher a fond les representation des smiles via des articles pour lencodage et 
la tokenisation 

Mettre lien vers article
mieux rediger read me
implementer plusieurs modele generatif
faire benchmark performance
interface
rechercher meilleur embedding

regarder RDKit pour validite des molecules
RDKit potentiel tokenizer


pipeline pour semaine du 28/07/2025
regarder representation smile et lire article
regarder VAE et lire article plus rafraichier memoire VAE utilisation
faire schéma explicatif pour mieux comprendre et pas hesiter à utiliser l outil de tensorflow tensorflow playground pour model
regarder metrique d evaluation



# Laisser de coté 

Méthode combinatoire y revenir plus tard

# installer pytorch