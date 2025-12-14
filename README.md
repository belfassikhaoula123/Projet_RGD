# Projet_RGD

## Classification et Segmentation de Tumeurs Cérébrales à partir d’IRM (Deep Learning)

Ce projet implémente un **pipeline Deep Learning complet** pour l’analyse d’images IRM cérébrales, avec :

* **Classification binaire** : tumeur / pas de tumeur (**CNN baseline + ResNet50 Transfer Learning**)
* **Segmentation** : localisation de la tumeur via des modèles (**U-Net classique + ResNet50-U-Net**)

## Données

Dataset dans `Data/kaggle_3m/` :

* Images IRM : `..._<n>.tif`
* Masques : `..._<n>_mask.tif`

### Labels (classification)

Le label est construit automatiquement à partir du masque :

* `0` : masque vide
* `1` : masque non vide

**Dataset final**

* 3929 images
* 2556 sans tumeur / 1373 avec tumeur

---

## 1) Classification (tumeur vs non-tumeur)

### Prétraitement & Data Augmentation

* Redimensionnement : **256×256**
* Normalisation : `rescale = 1./255`
* Augmentations (train) : rotation, shift, shear, zoom, flip horizontal

**Split**

* Train : 3182
* Validation : 354
* Test : 393

### Modèles de classification

#### 1.1 CNN (baseline)

Réseau convolutionnel simple :

* `Conv2D + MaxPooling2D`
* `Dense` + sortie binaire `sigmoid`

#### 1.2 ResNet50 (Transfer Learning)

Utilisation de **ResNet50 pré-entraîné sur ImageNet** :

* `include_top=False`
* ajout d’une tête de classification binaire (GlobalPooling + Dense + sigmoid)
* gel partiel des couches (fine-tuning partiel)

Objectif : extraire des caractéristiques plus robustes et améliorer la performance.

---

## 2) Segmentation (localisation de la tumeur)

### Filtrage du dataset

La segmentation est effectuée uniquement sur les images **avec tumeur** (`mask = 1`) :

* 1373 images

**Split**

* Train : 1167
* Validation : 103
* Test : 103

### Architectures

#### 2.1 ResNet50-U-Net

* **Encodeur** : ResNet50 (ImageNet, `include_top=False`)
* **Skip connections** : `conv1_relu`, `conv2_block3_out`, `conv3_block4_out`, `conv4_block6_out`
* **Bottleneck** : `conv5_block3_out`
* **Décodeur** : UpSampling2D + concaténation + blocs convolutionnels
* **Sortie** : `Conv2D(1, 1, activation="sigmoid")`

#### 2.2 U-Net classique

* **Encodeur** : Conv2D + BatchNorm + ReLU + MaxPooling (4 niveaux)
  Filtres : `64 → 128 → 256 → 512`
* **Bottleneck** : `2 × Conv2D(1024)`
* **Décodeur** : UpSampling2D + concaténation (4 niveaux)
  Filtres : `512 → 256 → 128 → 64`
* **Sortie** : masque binaire (sigmoid)

### Entraînement & métrique

* Loss : **Focal Tversky Loss**
* Métrique : **Tversky Index**
* Callbacks : EarlyStopping, ReduceLROnPlateau

---

## 3) Pipeline combiné : Classification + Segmentation

Pipeline en **2 étapes** :

1. **Classification**

* Prédit la probabilité de présence d’une tumeur (ResNet50)
* Si aucune tumeur n’est détectée, la segmentation est ignorée

2. **Segmentation conditionnelle**

* Si tumeur détectée, production d’un masque de segmentation (U-Net)


