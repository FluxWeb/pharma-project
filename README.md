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


# TO DO::


Mettre lien vers article
mieux rediger read me
implementer plusieurs modele generatif
faire benchmark performance
interface