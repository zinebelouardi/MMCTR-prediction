
#  Multimodal CTR Prediction System

##  Overview
Ce projet propose une **architecture neuronale multimodale hiérarchique** pour la prédiction du **Click-Through Rate (CTR)**.

L'objectif est de capturer :
- les interactions complexes entre utilisateurs et contenu
- les dépendances séquentielles
- les relations croisées entre features

---

##  Architecture globale

![Architecture](images/archi.PNG)

---

#  Task 1: Multimodal Embedding Extraction

##  Objectif
Transformer différentes modalités (texte, image, metadata) en **représentations vectorielles riches (embeddings)**.

##  Pipeline

### 1. Encodage multimodal
- Utilisation de **SigLIP** pour encoder :
  -  Images
  -  Texte

 Produit des embeddings alignés dans un même espace vectoriel.

---

### 2. Encodage des features contextuelles
Les données suivantes sont transformées en embeddings :
- tags
- vues
- likes

---

### 3. Modélisation séquentielle
Les embeddings sont injectés dans un **Transformer** :

- Capture les dépendances temporelles
- Modélise le comportement utilisateur
- Apprend les interactions séquentielles

---

##  Output Task 1
Un vecteur enrichi combinant :
- embeddings multimodaux
- contexte utilisateur
- historique séquentiel

---

#  Task 2: CTR Prediction

##  Objectif
Prédire la probabilité de clic (CTR) à partir des représentations apprises.

---

##  Pipeline

### 1. Fusion des features
Concaténation de :
- embeddings du Transformer
- features contextuelles

---

### 2. Apprentissage des interactions

####  DCNv2 (Deep & Cross Network)
- Capture les interactions **explicites**
- Apprend les relations croisées entre features

####  DNN (Deep Neural Network)
- Capture les interactions **implicites**
- Apprentissage non linéaire

---

### 3. Prédiction finale

- Fusion DCNv2 + DNN
- Passage dans un **MLP**
- Output : probabilité de clic

---

##  Output Task 2

-  CTR prediction (probabilité)
-  optimisation de la précision

---

##  Points forts

-  Architecture multimodale (image + texte)
-  Modélisation séquentielle (Transformer)
-  Feature crossing avancé (DCNv2)
-  Fusion hybride (explicite + implicite)
-  Pipeline end-to-end

---
##  Résultats

![Logo du projet](images/score.PNG)


