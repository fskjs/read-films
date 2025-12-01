# BiGRUmodel.py
import torch
import torch.nn as nn

class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_dimension = hidden_size * 2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        if lengths is not None:
            lengths = lengths.cpu()
            sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
            x_sorted = x[sorted_idx]

            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, sorted_lengths, batch_first=True, enforce_sorted=True)

            packed_output, hidden = self.gru(packed)

            hidden = torch.cat([
                hidden[-2, :, :],
                hidden[-1, :, :]
            ], dim=1)  # [batch, hidden_size*2]

            _, unsorted_idx = torch.sort(sorted_idx)
            hidden = hidden[unsorted_idx]

        else:
            output, hidden = self.gru(x)
            hidden = torch.cat([
                hidden[-2, :, :],
                hidden[-1, :, :]
            ], dim=1)  # [batch, hidden_size*2]

        return self.dropout(hidden)
