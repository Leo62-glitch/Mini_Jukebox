# Jukebox Light - Lightweight OpenAI Jukebox Model

Jukebox Light is a lightweight and modular version of the OpenAI Jukebox model, enabling music generation from text or audio prompts with reduced hardware requirements.  
This project aims to make Jukebox easier to use and experiment with by reducing the model size and VRAM consumption.

## Features

- Generate music from text or short audio prompts
- Lightweight model (smaller size, fewer parameters)
- Simple interface (CLI or Python script)
- Compatible with CPU and GPU (CUDA recommended for better performance)
- Ability to train on your own datasets
- Available on Google Colab: [Mini_Jukebox Notebook](https://colab.research.google.com/drive/1yaNAm3H4Gr4ZL7q2yDtgHPUZVgIfK91N?usp=sharing)
- Available on Hugging Face: [Mini_Jukebox HF Spaces](https://huggingface.co/spaces/Leo71288/Mini_Jukebox)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Leo62-glitch/Mini_Jukebox.git
   cd Mini_Jukebox
   git lfs install
   git lfs fetch
   git lfs checkout
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

### Music Generation

Command line example:
```bash
python ./setup.py generate
```

### Training on Your Own Data

1. Place your audio files in `./data/` in wav format
2. Start training:
   ```bash
   python ./setup.py train
   ```

## Project Structure for Generation

```
V2 Mini_Jukebox/
├── setup.py             # Python entry point
├── prior.pth            # Prior model
├── vqvae.pth            # VQ-VAE model
├── codes_simple.pth     # Extracted training audio data
├── requirements.txt     # Python dependencies
└── README.md
```

## Credits

- Based on [OpenAI Jukebox](https://github.com/openai/jukebox)

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

Feel free to open an issue for any questions or suggestions!
