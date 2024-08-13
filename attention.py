import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim

        # Linear layers to project the encoder and decoder outputs
        self.query = nn.Linear(decoder_dim, attention_dim * num_heads)
        self.key = nn.Linear(encoder_dim, attention_dim * num_heads)
        self.value = nn.Linear(encoder_dim, attention_dim * num_heads)

        self.fc_out = nn.Linear(attention_dim * num_heads, encoder_dim)

        self.dropout = nn.Dropout(0.1)

        self.scale = torch.sqrt(torch.FloatTensor([attention_dim])).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def forward(self, encoder_outputs, decoder_hidden):
        batch_size = encoder_outputs.size(0)

        # Linear projection
        queries = self.query(decoder_hidden)  # (batch_size, attention_dim * num_heads)
        keys = self.key(
            encoder_outputs
        )  # (batch_size, num_features, attention_dim * num_heads)
        values = self.value(
            encoder_outputs
        )  # (batch_size, num_features, attention_dim * num_heads)

        # Reshape for multi-head attention
        queries = queries.view(
            batch_size, self.num_heads, self.attention_dim
        )  # (batch_size, num_heads, attention_dim)
        keys = keys.view(batch_size, -1, self.num_heads, self.attention_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, num_features, attention_dim)
        values = values.view(
            batch_size, -1, self.num_heads, self.attention_dim
        ).transpose(
            1, 2
        )  # (batch_size, num_heads, num_features, attention_dim)

        # Scaled Dot-Product Attention
        energy = (
            torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        )  # (batch_size, num_heads, 1, num_features)
        attention = F.softmax(
            energy, dim=-1
        )  # (batch_size, num_heads, 1, num_features)
        attention = self.dropout(attention)

        out = torch.matmul(
            attention, values
        )  # (batch_size, num_heads, 1, attention_dim)
        out = (
            out.transpose(1, 2).contiguous().view(batch_size, -1)
        )  # (batch_size, num_heads * attention_dim)

        out = self.fc_out(out)  # (batch_size, encoder_dim)

        return out, attention.view(batch_size, self.num_heads, -1).mean(
            dim=1
        )  # (batch_size, num_features)


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return alpha, attention_weighted_encoding
