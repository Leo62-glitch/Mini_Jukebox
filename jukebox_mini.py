import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader

# -------- Dataset --------
class MusicDataset(Dataset):
    def __init__(self, folder='data', max_len=66000):
        self.files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        self.folder = folder
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        wav, sr = torchaudio.load(os.path.join(self.folder, f))
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)(wav)
        wav = wav.mean(dim=0).unsqueeze(0)  # mono
        if wav.size(1) > self.max_len:
            wav = wav[:, :self.max_len]
        else:
            pad = self.max_len - wav.size(1)
            wav = nn.functional.pad(wav, (0, pad))
        return wav

# -------- Vector Quantizer --------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()  # (B, T, C)
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (flat_input.pow(2).sum(1, keepdim=True)
                     - 2 * flat_input @ self.embeddings.weight.t()
                     + self.embeddings.weight.pow(2).sum(1))
        encoding_indices = distances.argmin(1)
        encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view(inputs.shape)
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()
        encoding_indices = encoding_indices.view(inputs.shape[0], inputs.shape[1])
        return quantized, loss, encoding_indices

# -------- Encoder --------
class Encoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Conv1d(256, embedding_dim, 4, 2, 1),
        )
    def forward(self, x):
        return self.net(x)

# -------- Decoder --------
class Decoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose1d(128, 1, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

# -------- VQ-VAE --------
class VQVAE(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=512):
        super().__init__()
        self.encoder = Encoder(embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, indices

# -------- Prior Transformer --------
class SimpleTransformer(nn.Module):
    def __init__(self, num_tokens, embedding_dim=64, n_layers=4, n_heads=4, max_len=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.token_emb = nn.Embedding(num_tokens, embedding_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, embedding_dim))  # Position embedding fixed here
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.to_logits = nn.Linear(embedding_dim, num_tokens)

    def forward(self, x):
        b, t = x.shape
        assert t <= self.pos_emb.size(1), "Sequence length exceeds max_len"

        x = self.token_emb(x) + self.pos_emb[:, :t, :]  # add positional embedding
        x = x.permute(1, 0, 2)  # (seq_len, batch, embedding_dim)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # back to (batch, seq_len, embedding_dim)
        logits = self.to_logits(x)
        return logits

    def generate(self, start_tokens, max_len=256):
        self.eval()
        tokens = start_tokens
        for _ in range(max_len - tokens.size(1)):
            logits = self.forward(tokens)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens

# -------- Training VQ-VAE --------
def train_vqvae(device, epochs=10):
    dataset = MusicDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = VQVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, vq_loss, _ = model(x)
            recon_loss = nn.functional.mse_loss(x_hat, x)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[VQ-VAE] Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), "vqvae.pth")

# -------- Extract codes (sans conditions) --------
def extract_codes_simple(device):
    dataset = MusicDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = VQVAE().to(device)
    model.load_state_dict(torch.load("vqvae.pth", map_location=device))
    model.eval()
    codes = []
    for x in dataloader:
        x = x.to(device)
        with torch.no_grad():
            _, _, indices = model(x)
        codes.append(indices.squeeze(0).cpu())
    torch.save(codes, "codes_simple.pth")
    print("Codes saved.")

# -------- Train prior sur codes seuls --------
def train_prior_simple(device, epochs=5):
    codes = torch.load("codes_simple.pth")
    max_len = max(c.size(0) for c in codes)
    vocab_size = 512
    model = SimpleTransformer(num_tokens=vocab_size, max_len=30000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    data = []
    for c in codes:
        if c.size(0) < max_len:
            pad_len = max_len - c.size(0)
            c = torch.cat([c, torch.zeros(pad_len, dtype=torch.long)])
        data.append(c)

    class CodesDataset(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx]

    dataloader = DataLoader(CodesDataset(data), batch_size=2, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            inp = batch[:, :-1]
            target = batch[:, 1:]
            optimizer.zero_grad()
            out = model(inp)
            loss = nn.functional.cross_entropy(out.reshape(-1, vocab_size), target.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Prior] Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), "prior.pth")

# -------- Génération --------
def generate_simple(device, max_len=30000):
    vocab_size = 512
    model_prior = SimpleTransformer(num_tokens=vocab_size, max_len=max_len).to(device)  # <-- préciser max_len ici
    model_prior.load_state_dict(torch.load("./prior.pth", map_location=device))
    model_prior.eval()

    start_tokens = torch.zeros((1, 1), dtype=torch.long).to(device)  # start from token 0

    generated_codes = model_prior.generate(start_tokens, max_len=max_len).squeeze(0)

    vqvae = VQVAE().to(device)
    vqvae.load_state_dict(torch.load("./vqvae.pth", map_location=device))
    vqvae.eval()

    embeddings = vqvae.vq.embeddings.weight  # (num_embeddings, embedding_dim)
    quantized = embeddings[generated_codes].permute(1, 0).unsqueeze(0)  # (1, C, T)
    with torch.no_grad():
        audio = vqvae.decoder(quantized)
    audio = audio.squeeze().cpu()
    torchaudio.save("generated_simple.wav", audio.unsqueeze(0), 22050)
    print("Audio généré sauvegardé dans generated_simple.wav")

# -------- CLI simplifié --------
if __name__ == "__main__":
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd == "train_vqvae":
        train_vqvae(device)
    elif cmd == "extract_codes":
        extract_codes_simple(device)
    elif cmd == "train_prior":
        train_prior_simple(device)
    elif cmd == "generate":
        generate_simple(device)
    else:
        print("Commandes: train_vqvae | extract_codes | train_prior | generate")
