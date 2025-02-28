from datasets import load_dataset
from transformers import BertTokenizer

def load_translation_dataset(max_length=32):
    """
    Loads the WMT14 English-German translation test dataset,
    initializes the tokenizers, and tokenizes the data.
    """
    # Load dataset (using test split for demonstration)
    dataset = load_dataset('wmt14', 'de-en', split='test')
    
    # Initialize tokenizers
    src_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # English
    tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')  # German

    def tokenize_data(batch):
        src = src_tokenizer(batch['translation']['en'], padding="max_length", truncation=True, max_length=max_length)
        tgt = tgt_tokenizer(batch['translation']['de'], padding="max_length", truncation=True, max_length=max_length)
        return {'src_input_ids': src['input_ids'], 'tgt_input_ids': tgt['input_ids']}

    # Tokenize and pad the dataset
    dataset = dataset.map(tokenize_data)
    return dataset, src_tokenizer, tgt_tokenizer