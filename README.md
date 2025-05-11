# Transformer from scratch

Transformers have revolutionized the field of Natural Language Processing (NLP) by introducing a novel mechanism for capturing dependencies within sequences through attention mechanisms. 

This project implements the Transformer model from scratch using PyTorch, based on the paper: [*Attention Is All You Need!*](https://arxiv.org/abs/1706.03762).

**Note**: All implementations and analysis are located in `notebooks/transformer-from-scratch.ipynb.`

##  Features

- Implements a full Transformer architecture with encoder and decoder blocks
- Includes multi-head attention, positional encoding, feedforward networks, and residual connections
- Trains on the WMT14 English-German translation dataset (test split used for demonstration)
- Customizable hyperparameters for experimentation

##  Results
### Training Log Summary
| Epoch | Train Loss | Validation Loss |
|-------|-----------|----------------|
| 1     | 0.3104    | 0.3037         |
| 2     | 0.2901    | 0.2873         |
| 3     | 0.2730    | 0.2742         |
| 4     | 0.2595    | 0.2642         |
| 5     | 0.2491    | 0.2567         |
| 6     | 0.2415    | 0.2518         |
| 7     | 0.2363    | 0.2489         |
| 8     | 0.2331    | 0.2473         |
| 9     | 0.2309    | 0.2462         |
| 10    | 0.2294    | 0.2456         |

🔹 **Final Validation Loss:** `0.2456`  
🔹 **Training Log:** Available in [`results/train_logs.txt`](results/train_logs.txt)

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- PyTorch Documentation: [MultiheadAttention module](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html), [Transformer module](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- WMT14 English-German Dataset: [https://www.statmt.org/wmt14/translation-task.html](https://www.statmt.org/wmt14/translation-task.html)

