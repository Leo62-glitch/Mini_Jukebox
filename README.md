# Jukebox Light - Modèle Léger OpenAI Jukebox

Jukebox Light est une version allégée et modulaire du modèle OpenAI Jukebox, permettant de générer de la musique à partir de texte ou de prompts audio, avec des ressources matérielles réduites.  
Ce projet vise à faciliter l’utilisation et l’expérimentation avec Jukebox, en réduisant la taille du modèle et les besoins en VRAM.

## Fonctionnalités

- Génération musicale à partir de textes ou d’audio courts
- Modèle allégé (taille réduite, moins de paramètres)
- Interface simple d’utilisation (CLI ou script Python)
- Compatible CPU et GPU (CUDA conseillé pour de meilleures performances)
- Possibilité d’entraîner sur vos propres datasets

## Installation

1. Clonez le dépôt :
   ```bash
   Télécharger ce dépo et extraire le .zip
   cd Mini_Jukebox
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Télécharger les modèles sur hugging face [Mini Jukebox HF](https://huggingface.co/Leo71288/OpenAI_Mini-Jukebox) et déplacer les modèles dans Mini_Jukebox

## Utilisation

### Génération de musique

Exemple en ligne de commande :
```bash
python ./setup.py generate
```

### Entraînement sur vos propres données

1. Placez vos fichiers audio dans `./data/`
2. Lancez l’entraînement :
   ```bash
   python ./setup.py train_vqvae
   python ./setup.py train_prior
   python ./setup.py extract_codes
   ```

## Arborescence du projet pour générer

```
V2 Mini Jukebox/
├── setup.py      # Code .py
├── prior.pth             # Modèle Prior
├── vqvae.pth       # Modèle VQ-VAE
├── codes_simple.pth          # Extraction des données audio d'entrainement
├── requirements.txt  # Dépendances Python
└── README.md
```

## Crédits

- Basé sur [OpenAI Jukebox](https://github.com/openai/jukebox)

## License

Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus d’informations.

---

N’hésitez pas à ouvrir une issue pour toute question ou suggestion !
