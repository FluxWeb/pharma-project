# molecularb generation project


### creation d environnement 
python -m venv mol_gen


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


## datasets

MOSES : https://github.com/molecularsets/moses