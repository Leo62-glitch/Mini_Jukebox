# Jukebox Light - Lightweight OpenAI Jukebox Model

Jukebox Light is a lightweight and modular version of the OpenAI Jukebox model, enabling music generation from text or audio prompts with reduced hardware resources.  
This project aims to make it easier to use and experiment with Jukebox by reducing the model size and VRAM requirements.

## Features

- Music generation from text or short audio prompts
- Lightweight model (smaller size, fewer parameters)
- Simple interface (CLI or Python script)
- Compatible with CPU and GPU (CUDA recommended for better performance)
- Ability to train on your own datasets
- Available on Google Colab: [https://colab.research.google.com/drive/1yaNAm3H4Gr4ZL7q2yDtgHPUZVgIfK91N?usp=sharing](Mini_Jukebox Notebook)

## Installation

1. Clone the repository:
   ```bash
   Download this repo and extract the .zip
   cd Mini_Jukebox
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the models from Hugging Face [Mini Jukebox HF](https://huggingface.co/Leo71288/OpenAI_Mini-Jukebox) and move them into Mini_Jukebox

## Usage

### Music Generation

Command line example:
```bash
python ./setup.py generate
```

### Training on Your Own Data

1. Place your audio files in `./data/`
2. Start training:
   ```bash
   python ./setup.py train_vqvae
   python ./setup.py train_prior
   python ./setup.py extract_codes
   ```

## Project Structure for Generation

```
V2 Mini Jukebox/
├── setup.py      # Python code
├── prior.pth             # Prior model
├── vqvae.pth       # VQ-VAE model
├── codes_simple.pth          # Extracted training audio data
├── requirements.txt  # Python dependencies
└── README.md
```

## Credits

- Based on [OpenAI Jukebox](https://github.com/openai/jukebox)

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

Feel free to open an issue for any questions or suggestions!
