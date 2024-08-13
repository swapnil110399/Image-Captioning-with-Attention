import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import *


class GRUDecoder(nn.Module):
    def __init__(
        self,
        embed_size,
        vocab_size,
        attention_dim,
        encoder_dim,
        decoder_dim,
        num_heads=8,
        drop_prob=0.3,
    ):
        super(GRUDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.num_heads = num_heads

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = MultiHeadAttention(
            encoder_dim, decoder_dim, attention_dim, num_heads
        )

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.gru_cell = nn.GRUCell(embed_size + encoder_dim, decoder_dim)

        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(drop_prob)
        self.layer_norm = nn.LayerNorm(decoder_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)

    def forward(self, encoder_outputs, captions):
        print("here")
        embeddings = self.embedding(captions)
        h = self._init_hidden_state(encoder_outputs)

        batch_size = captions.size(0)
        seq_length = captions.size(1) - 1
        num_features = encoder_outputs.size(1)

        outputs = torch.zeros(batch_size, seq_length, self.vocab_size).to(
            encoder_outputs.device
        )
        attention_weights = torch.zeros(batch_size, seq_length, num_features).to(
            encoder_outputs.device
        )

        for t in range(seq_length):
            attention_out, attn_weights = self.attention(encoder_outputs, h, None)
            gru_input = torch.cat([embeddings[:, t], attention_out], dim=1)
            h = self.gru_cell(gru_input, h)
            h = self.layer_norm(h)

            output = self.fc(self.dropout(h))
            outputs[:, t] = output
            attention_weights[:, t] = attn_weights

        return outputs, attention_weights

    def generate_caption(self, encoder_outputs, max_length=20, vocab=None):
        batch_size = encoder_outputs.size(0)
        h = self._init_hidden_state(encoder_outputs)

        alphas = []
        captions = []

        start_token = torch.tensor([vocab.stoi["<SOS>"]] * batch_size).to(
            encoder_outputs.device
        )
        current_word = self.embedding(start_token)

        for _ in range(max_length):
            attention_out, attn_weights = self.attention(encoder_outputs, h)
            alphas.append(attn_weights.cpu().detach().numpy())

            gru_input = torch.cat([current_word.squeeze(1), attention_out], dim=1)
            h = self.gru_cell(gru_input, h)
            h = self.layer_norm(h)

            output = self.fc(self.dropout(h))
            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.tolist())

            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break

            current_word = self.embedding(predicted_word_idx.unsqueeze(1))

        caption_words = [vocab.itos[idx] for idx in captions[0]]
        return caption_words, alphas

    def _init_hidden_state(self, encoder_outputs):
        mean_encoder_out = encoder_outputs.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        return h


class DecoderRNN(nn.Module):
    def __init__(
        self,
        embed_size,
        vocab_size,
        attention_dim,
        encoder_dim,
        decoder_dim,
        drop_prob=0.3,
    ):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)
        self.layer_norm = nn.LayerNorm(decoder_dim)

        self.init_weights()

    def init_weights(self):
        """Initialize weights for embedding and linear layers"""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fcn.bias.data.fill_(0)
        self.fcn.weight.data.uniform_(-0.1, 0.1)

    def forward(self, features, captions):
        embeds = self.embedding(captions)

        h, c = self.init_hidden_state(features)

        seq_length = captions.size(1) - 1
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(features.device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(features.device)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            h = self.layer_norm(h)

            output = self.fcn(self.drop(h))
            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas

    def generate_caption(self, features, max_len=20, vocab=None):
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)

        alphas = []

        word = torch.tensor(vocab.stoi["<SOS>"]).view(1, -1).to(features.device)
        embeds = self.embedding(word)

        captions = []

        for i in range(max_len):
            alpha, context = self.attention(features, h)
            alphas.append(alpha.cpu().detach().numpy())

            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            h = self.layer_norm(h)
            output = self.fcn(self.drop(h))
            output = output.view(batch_size, -1)

            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.item())

            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break

            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        return [vocab.itos[idx] for idx in captions], alphas

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
