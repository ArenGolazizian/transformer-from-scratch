import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

def create_src_mask(src_input, pad_idx=0):
    return (src_input != pad_idx).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt_input, pad_idx=0):
    batch_size, tgt_len = tgt_input.shape
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool().to(tgt_input.device).unsqueeze(0)
    pad_mask = (tgt_input != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_mask = causal_mask & pad_mask.squeeze(1)
    return tgt_mask.unsqueeze(1)

def evaluate_transformer(transformer, dataset, src_tokenizer, tgt_tokenizer, device, batch_size=32):
    transformer.eval()
    total_loss = 0
    criterion = CrossEntropyLoss(ignore_index=0)
    tgt_vocab_size = tgt_tokenizer.vocab_size

    # Evaluate on dataset from index 2000 to end
    for i in tqdm(range(2000, len(dataset), batch_size), desc="Evaluating"):
        src_input = torch.tensor(dataset[i:i+batch_size]['src_input_ids']).to(device)
        tgt_input = torch.tensor(dataset[i:i+batch_size]['tgt_input_ids']).to(device)

        src_mask = create_src_mask(src_input)
        tgt_mask = create_tgt_mask(tgt_input[:, :-1])
        with torch.no_grad():
            encoder_output = transformer.encode(src_input, src_mask)
            decoder_output = transformer.decode(encoder_output, src_mask, tgt_input[:, :-1], tgt_mask)
            output = transformer.project(decoder_output)
            loss = criterion(output.view(-1, tgt_vocab_size), tgt_input[:, 1:].reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / ((len(dataset) - 2000) // batch_size)
    print(f'Average Evaluation Loss: {avg_loss}')

if __name__ == "__main__":
    from src.dataset import load_translation_dataset
    from src.model import build_transformer
    from transformers import BertTokenizer
    dataset, src_tokenizer, tgt_tokenizer = load_translation_dataset(max_length=32)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer = build_transformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        src_seq_len=32,
        tgt_seq_len=32
    ).to(device)
    
    # Load your trained model weights here if saved, for example:
    # transformer.load_state_dict(torch.load("results/transformer_model.pth"))
    
    evaluate_transformer(transformer, dataset, src_tokenizer, tgt_tokenizer, device, batch_size=32)
