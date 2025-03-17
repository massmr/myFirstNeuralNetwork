## Description  
Ce projet implémente un réseau de neurones multicouche en **PyTorch** pour une tâche de **classification binaire**.  
Le modèle apprend à classifier des points générés aléatoirement selon une règle simple : la somme des coordonnées doit être supérieure à 1 pour être classée comme 1, sinon 0.  

## Installation  
### Prérequis  
- Python 3.x  
- PyTorch  
- NumPy  

### Installation des dépendances  
Assurez-vous que PyTorch est installé sur votre machine. Si ce n'est pas le cas, installez-le avec :  
```bash
pip install torch numpy
```

## Utilisation  
### Exécution du script  
Lancez simplement le script Python :  
```bash
python neural_network.py
```
Cela entraînera le modèle pendant 50 époques et affichera la perte toutes les 10 époques, ainsi que la précision finale sur un ensemble de test.  

## Explication du Code  
### 1. Génération des données  
Le jeu de données est généré de manière aléatoire avec des points `(X1, X2)` entre 0 et 1.  
L'étiquette `y` est définie comme :  
- `y = 1` si `X1 + X2 > 1`  
- `y = 0` sinon  

### 2. Définition du Modèle  
Le réseau est composé de **trois couches** :  
- Une couche cachée de **6 neurones** avec activation **ReLU**  
- Une seconde couche cachée de **4 neurones** avec **ReLU**  
- Une couche de sortie de **1 neurone** avec **sigmoïde** pour obtenir une probabilité entre 0 et 1  

### 3. Fonction coût et optimisation  
- La **fonction coût** utilisée est l'**entropie croisée binaire** (`BCELoss`).  
- L'**optimiseur** est **Adam** avec un taux d'apprentissage de `0.01`.  

### 4. Entraînement du modèle  
Le réseau est entraîné pendant `50` époques en utilisant la **descente de gradient** avec rétropropagation.  

### 5. Évaluation  
Après l'entraînement, le modèle est testé sur un ensemble de test et la précision est calculée.  

## Explication Mathématique  
### 1. Fonction d'activation  
- **ReLU** :  
  ```
  ReLU(x) = max(0, x)
  ```
  Utilisée pour éviter le problème de la **vanishing gradient**.  
- **Sigmoïde** :  
  ```
  σ(x) = 1 / (1 + e^(-x))
  ```
  Utilisée en sortie pour convertir la sortie en probabilité.  

### 2. Fonction Coût (Binary Cross Entropy)  
```
L = - (1/N) * Σ [ y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i) ]
```
avec :  
- `y_i` : vrai label (`0` ou `1`)  
- `ŷ_i` : prédiction du modèle  

## Résultats Attendus
Après l'entraînement, la précision sur les données de test devrait être d'environ **80-90%**.


