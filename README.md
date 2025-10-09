# molecularb generation project


necessite python 3.12 ou moins pour installer torch avec CUDA

j utilise la version 3.11

compiler avec la version python 
py -3.11 mon_script.py

mettre à jour pip

### creation d environnement 

Windows:
py -3.11 -m venv mol_gen

Ubuntu:
python3 -m venv ~/mol_gen

### activer environnement


Windows:

In cmd.exe pas par default dans visual studio 
 mol_gen\Scripts\activate.bat

note l espace est important

In PowerShell (attention il faut definir une strategie d execution python)
mol_gen\Scripts\Activate.ps1

Ubuntu:

source ~/mol_gen/bin/activate

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
requirements a faire

# installer pytorch


# materiel

https://developers.google.com/machine-learning/crash-course/embeddings?hl=fr
explication embedding
chemberta represente chaque token sur un vecteur 1x768 


# Arborescence a mettre en place

## Lien pour la structure

https://medium.com/%40marameref/building-a-simple-and-professional-mlops-project-structure-a-hands-on-project-5facd53c9268

https://dev.to/luxdevhq/generic-folder-structure-for-your-machine-learning-projects-4coe?utm_source=chatgpt.com

https://santiagof.medium.com/structure-your-machine-learning-project-source-code-like-a-pro-44815cac8652

https://dagster.io/blog/python-project-best-practices?utm_source=chatgpt.com

## Structure 

mon_projet_ml/
├── bin/                  # Exécutables C++ compilés (Qt UI)
├── build/                # Dossiers de build (CMake)
├── cmake/                # Fichiers de configuration CMake
├── data/                 # Données
│   ├── raw/
│   ├── processed/
├── include/              # Headers C++ (.h / .hpp)
├── lib/                  # Bibliothèques C++ externes compilées
├── python/               # Scripts Python ML
│   ├── model.py          # Modèle ML
│   └── utils.py          # Fonctions utilitaires
├── scripts/              # Scripts shell pour lancer l’UI ou les tests
├── src/                  # Code source C++ (Qt interface, logique)
│   ├── main.cpp
│   ├── ui/               # Widgets, fenêtres
│   └── ml_interface/     # Code pour appeler Python depuis C++
├── tests/                # Tests C++ et Python
├── CMakeLists.txt        # Fichier CMake principal
└── README.md
