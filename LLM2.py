import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader

# Define a simple tokenizer using Byte Pair Encoding (BPE)
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer (or load pretrained tokenizer if available)
# Make sure to train it on your dataset (example) before use
# tokenizer.train_from_file('path_to_your_text_corpus.txt', vocab_size=5000)

# Sample dataset for training (replace with actual dataset later)
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        
        # Encode texts
        self.encoded_texts = [self.tokenizer.encode(text).ids for text in texts]
        
        # Pad sequences to max_length
        self.encoded_texts = [x[:max_length] + [0] * (max_length - len(x)) for x in self.encoded_texts]
    
    def __len__(self):
        return len(self.encoded_texts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.encoded_texts[idx], dtype=torch.long)

# Sample dataset (small test corpus)
texts = ["Hello world!", "MiniGPT is learning.", "Transformers are powerful."]
train_dataset = TextDataset(texts, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, num_layers=2, max_seq_len=128):
        super(MiniGPT, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.transformer = nn.Transformer(d_model, n_heads, num_layers, num_layers, dim_feedforward=256, batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        if x.size(1) == 0:  # Handle empty input case
            return torch.zeros((x.size(0), 1, self.fc_out.out_features), device=x.device)
        x = self.embed(x) + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x, x)  # Decoder-only model
        return self.fc_out(x)

# Training setup
def train_model(model, dataloader, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output.view(-1, output.size(-1)), batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            loss = loss_fn(output.view(-1, output.size(-1)), batch.view(-1))
            total_loss += loss.item()
    print(f"Evaluation Loss: {total_loss / len(dataloader):.4f}")
    model.train()

# Text generation function with temperature and top-k sampling
def generate_text(model, tokenizer, seed_text, max_length=20, temperature=1.0, top_k=5):
    model.eval()
    
    # Tokenize seed_text
    tokens = tokenizer.encode(seed_text).ids
    print(f"Tokenized input: {tokens}")  # Debug tokenization

    # Check if tokenized input is valid
    if len(tokens) == 0:
        return "[Error: Empty input tokenized]"

    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    
    for _ in range(max_length):
        with torch.no_grad():
            output = model(tokens)
            if output.size(1) == 0:  # If model produces no tokens, stop
                break
            
            logits = output[:, -1, :] / temperature  # Apply temperature scaling
            
            # Top-k filtering
            top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            probs = torch.softmax(top_k_values, dim=-1)
            next_token = top_k_indices[0, torch.multinomial(probs, 1)].item()

        tokens = torch.cat([tokens, torch.tensor([[next_token]])], dim=1)
        
        # Check if the generated token is a padding or EOS token (adjust based on tokenizer's specs)
        eos_token_id = tokenizer.token_to_id('<|endoftext|>')  # Adjust if needed
        if next_token == eos_token_id:  # Example for EOS token
            break
    
    generated_text = tokenizer.decode(tokens.squeeze().tolist())
    return generated_text

# Initialize model and train
vocab_size = 5000  # Small vocabulary for testing
model = MiniGPT(vocab_size)

# Train the model
train_model(model, train_loader)

# Evaluate the model
evaluate_model(model, train_loader)

# Generate text
seed_text = "MiniGPT is"
generated_output = generate_text(model, tokenizer, seed_text, temperature=0.8, top_k=5)
print("Generated Text:", generated_output)
