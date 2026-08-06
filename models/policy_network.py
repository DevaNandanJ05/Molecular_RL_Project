import torch
import torch.nn as nn
import torch.nn.functional as F

class SmilesTokenizer:
    def __init__(self):
        # Restricted vocabulary focused on stable organic drug-like atoms/bonds
        chars = ['<pad>', '<sos>', '<eos>', 'C', 'c', 'O', 'N', 'n', 'F', '(', ')', '=', '1', '2']
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        self.sos_idx = self.char_to_idx['<sos>']
        self.eos_idx = self.char_to_idx['<eos>']
        self.pad_idx = self.char_to_idx['<pad>']

    def encode(self, smiles):
        return [self.sos_idx] + [self.char_to_idx.get(c, self.pad_idx) for c in smiles] + [self.eos_idx]

    def decode(self, indices):
        smiles = ""
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx not in [self.sos_idx, self.pad_idx]:
                smiles += self.idx_to_char[idx]
        return smiles

class RNNSmilesGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256):
        super(RNNSmilesGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # x shape: (batch_size, 1)
        embedded = self.embedding(x)
        out, hidden = self.gru(embedded, hidden)
        logits = self.fc(out.squeeze(1))
        return logits, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)