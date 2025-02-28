import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from model import build_transformer
from datasets import load_dataset  # if needed for your training
from transformers import BertTokenizer

# Define mask functions
def create_src_mask(src_input, pad_idx=0):
    """Create a mask for the source to hide padding tokens."""
    return (src_input != pad_idx).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt_input, pad_idx=0):
    """Create a target mask to hide future tokens and padding tokens."""
    batch_size, tgt_len = tgt_input.shape
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool().to(tgt_input.device).unsqueeze(0)
    pad_mask = (tgt_input != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_mask = causal_mask & pad_mask.squeeze(1)
    return tgt_mask.unsqueeze(1)

def train_transformer(transformer, dataset, src_tokenizer, tgt_tokenizer, device, epochs=10, batch_size=32):
    # Set vocab sizes from tokenizers
    src_vocab_size = src_tokenizer.vocab_size
    tgt_vocab_size = tgt_tokenizer.vocab_size

    criterion = CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(transformer.parameters(), lr=2e-5)
    train_loss_total = 0

    for epoch in range(epochs):
        transformer.train()
        train_loss = 0
        # Here we assume dataset is indexable and batched by slicing
        for i in tqdm(range(0, 2000, batch_size), desc=f"Training Epoch {epoch+1}"):
            src_input = torch.tensor(dataset[i:i+batch_size]['src_input_ids']).to(device)
            tgt_input = torch.tensor(dataset[i:i+batch_size]['tgt_input_ids']).to(device)

            src_mask = create_src_mask(src_input)
            tgt_mask = create_tgt_mask(tgt_input[:, :-1])

            optimizer.zero_grad()
            encoder_output = transformer.encode(src_input, src_mask)
            decoder_output = transformer.decode(encoder_output, src_mask, tgt_input[:, :-1], tgt_mask)
            output = transformer.project(decoder_output)

            loss = criterion(output.view(-1, tgt_vocab_size), tgt_input[:, 1:].reshape(-1))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Train loss: {train_loss/(2000//batch_size)}')
        train_loss_total += train_loss

    return transformer

if __name__ == "__main__":
    # Load dataset and tokenizers (use the same function as in dataset.py or here for simplicity)
    from src.dataset import load_translation_dataset  # assuming you saved the function there
    dataset, src_tokenizer, tgt_tokenizer = load_translation_dataset(max_length=32)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer = build_transformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        src_seq_len=32,
        tgt_seq_len=32
    ).to(device)
    
    trained_transformer = train_transformer(transformer, dataset, src_tokenizer, tgt_tokenizer, device, epochs=10, batch_size=32)