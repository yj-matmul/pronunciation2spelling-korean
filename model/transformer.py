import numpy as np
import math
import torch
import torch.nn as nn


ACT2FN = {'gelu': nn.GELU(), 'relu': nn.ReLU()}


class TransformerConfig:
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 hidden_size=512,
                 num_hidden_layers=6,
                 num_attn_head=8,
                 hidden_act='gelu',
                 device='cuda',
                 feed_forward_size=2048,
                 padding_idx=0,
                 share_embeddings=False,
                 hidden_dropout_prob=0.1,
                 attn_dropout_prob=0.1,
                 enc_max_seq_length=512,
                 dec_max_seq_length=512,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attn_head = num_attn_head
        self.hidden_act = hidden_act
        self.device = device
        self.feed_forward_size = feed_forward_size
        self.padding_idx = padding_idx
        self.share_embeddings = share_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.enc_max_seq_length = enc_max_seq_length
        self.dec_max_seq_length = dec_max_seq_length
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


# sinusoid positional encoding
def get_positional_encoding_table(seq_length, hidden_size):
    def get_angle(position, idx_hidden):
        return position / np.power(10000, 2 * (idx_hidden // 2) / hidden_size)

    def get_position_angle_vector(position):
        return [get_angle(position, idx_hidden) for idx_hidden in range(hidden_size)]

    sinusoid_table = np.array([get_position_angle_vector(idx_seq) for idx_seq in range(seq_length)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table)


def get_pad_mask(q_ids, k_ids, padding_idx):
    q_seq_length = q_ids.size()[1]  # q_ids : [batch_size, q_seq_length]
    k_seq_length = k_ids.size()[1]  # k_dis : [batch_size, k_seq_length]

    # attn_mask : [batch_size, q_seq_length, k_seq_length]
    attn_mask = k_ids.eq(padding_idx).unsqueeze(1).expand(-1, q_seq_length, k_seq_length)
    return attn_mask


def get_look_ahead_mask(input_ids):
    seq_length = input_ids.size()[1]  # input_ids : [batch_size, seq_length]

    # look_ahead_mask : [batch_size, seq_length, seq_length]
    look_ahead_mask = torch.ones_like(input_ids).unsqueeze(1).expand(-1, seq_length, seq_length)
    look_ahead_mask = look_ahead_mask.triu(diagonal=1).bool()
    return look_ahead_mask


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.padding_idx = config.padding_idx
        self.device = config.device

        self.src_token_embeddings = nn.Embedding(config.src_vocab_size, config.hidden_size)
        if config.share_embeddings:
            self.trg_token_embeddings = nn.Embedding(config.trg_vocab_size, config.hidden_size)
        else:
            self.trg_token_embeddings = self.src_token_embeddings
        max_seq_length = max(config.enc_max_seq_length, config.dec_max_seq_length)
        position_table = get_positional_encoding_table(max_seq_length + 1, config.hidden_size)
        self.position_encodings = nn.Embedding.from_pretrained(position_table, freeze=True)

    def forward(self, enc_ids, dec_ids):
        batch_size, enc_seq_length = enc_ids.size()
        batch_size, dec_seq_length = dec_ids.size()

        enc_position_ids = torch.arange(enc_seq_length, dtype=torch.long, device=self.device)
        enc_position_ids = enc_position_ids.unsqueeze(0).expand_as(enc_ids) + 1
        position_mask = enc_ids.eq(self.padding_idx)
        # enc_position_ids : [batch_size, enc_seq_length]
        enc_position_ids = torch.masked_fill(enc_position_ids, position_mask, self.padding_idx)

        dec_position_ids = torch.arange(dec_seq_length, dtype=torch.long, device=self.device)
        dec_position_ids = dec_position_ids.unsqueeze(0).expand_as(dec_ids) + 1
        position_mask = dec_ids.eq(self.padding_idx)
        # dec_position_ids : [batch_size, dec_seq_length]
        dec_position_ids = torch.masked_fill(dec_position_ids, position_mask, self.padding_idx)

        # src_word_embeddings : [batch_size, src_seq_length, hidden_size]
        src_token_embeddings = self.src_token_embeddings(enc_ids)
        # trg_word_embeddings : [batch_size, trg_seq_length, hidden_size]
        trg_token_embeddings = self.trg_token_embeddings(dec_ids)

        enc_embeddings = src_token_embeddings + self.position_encodings(enc_position_ids)
        dec_embeddings = trg_token_embeddings + self.position_encodings(dec_position_ids)
        return enc_embeddings, dec_embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_attn_head = config.num_attn_head
        self.attn_head_size = config.hidden_size // config.num_attn_head
        self.all_head_size = self.num_attn_head * self.attn_head_size

        self.weight_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.weight_key = nn.Linear(config.hidden_size, self.all_head_size)
        self.weight_value = nn.Linear(config.hidden_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_attn_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attn_head, self.attn_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, pad_mask):
        # [batch_size, seq_length, hidden_size]
        mixed_query_layer = self.weight_query(query)
        mixed_key_layer = self.weight_query(key)
        mixed_value_layer = self.weight_value(value)

        # [batch_size, num_attn_heads, seq_length, attn_head_size]
        query_layer = self.transpose_for_attn_scores(mixed_query_layer)
        key_layer = self.transpose_for_attn_scores(mixed_key_layer)
        value_layer = self.transpose_for_attn_scores(mixed_value_layer)

        # [batch_size, num_attn_heads, seq_length, seq_length]
        attn_mask = pad_mask.unsqueeze(1).repeat(1, self.num_attn_head, 1, 1)

        # [batch_size, num_attn_heads, seq_length, seq_length]
        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.attn_head_size)
        attn_scores.masked_fill_(attn_mask, -1e9)

        # [batch_size, num_attn_heads, seq_length, seq_length]
        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_probs = self.attn_dropout(attn_probs)

        # [batch_size, num_attn_heads, seq_length, attn_head_size]
        context_layer = torch.matmul(attn_probs, value_layer).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        # [batch_size, seq_length, hidden_size]
        context_layer = context_layer.view(new_context_layer_shape)

        # [batch_size, seq_length, hidden_size]
        hidden_states = self.dense(context_layer)
        hidden_states = self.hidden_dropout(hidden_states)
        return hidden_states, attn_probs


class FeedForwardNet(nn.Module):
    def __init__(self, config):
        super(FeedForwardNet, self).__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.feed_forward_size)
        self.ffn_act = ACT2FN[config.hidden_act]
        self.dense2 = nn.Linear(config.feed_forward_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs):
        # hidden_states: [batch_size, seq_length, feed_forward_size]
        hidden_states = self.dense1(inputs)
        hidden_states = self.ffn_act(hidden_states)
        # hidden_states : [batch_size, seq_length, hidden_size]
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForwardNet(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, enc_inputs, pad_mask):
        # self_attn_outputs : [batch_size, seq_length, hidden_size],
        # attn_probs : [batch_size, num_attn_heads, seq_length, seq_length]
        self_attn_outputs, attn_probs = self.self_attn(enc_inputs, enc_inputs, enc_inputs, pad_mask)
        self_attn_outputs = self.layer_norm1(self_attn_outputs + enc_inputs)

        # ffn_outputs : [batch_size, seq_length, hidden_size]
        ffn_outputs = self.ffn(self_attn_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + self_attn_outputs)
        return ffn_outputs, attn_probs


class Encoders(nn.Module):
    def __init__(self, config):
        super(Encoders, self).__init__()
        self.padding_idx = config.padding_idx
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, enc_ids, enc_embeddings):
        # hidden_states : [batch_size, enc_seq_length, hidden_size]
        hidden_states = enc_embeddings
        # pad_mask : [batch_size, enc_seq_length, enc_seq_length]
        pad_mask = get_pad_mask(enc_ids, enc_ids, self.padding_idx)

        attn_probs = []
        for i, layer in enumerate(self.layers):
            hidden_states, attn_prob = layer(hidden_states, pad_mask)
            attn_probs.append(attn_prob)
        return hidden_states, attn_probs


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.enc_dec_attn = MultiHeadAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForwardNet(config)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, dec_inputs, enc_outputs, look_ahead_mask, pad_mask):
        # self_attn_outputs : [batch_size, dec_seq_length, hidden_size],
        # self_attn_probs : [batch_size, num_attn_heads, dec_seq_length, dec_seq_length]
        self_attn_outputs, self_attn_probs = self.self_attn(dec_inputs, dec_inputs, dec_inputs, look_ahead_mask)
        self_attn_outputs = self.layer_norm1(self_attn_outputs + dec_inputs)

        # attn_outputs : [batch_size, seq_length, hidden_size],
        # attn_probs : [batch_size, num_attn_heads, dec_seq_length, enc_seq_length]
        attn_outputs, attn_probs = self.enc_dec_attn(self_attn_outputs, enc_outputs, enc_outputs, pad_mask)
        attn_outputs = self.layer_norm2(attn_outputs + self_attn_outputs)

        # ffn_outputs : [batch_size, dec_seq_length, hidden_size]
        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.layer_norm3(ffn_outputs + attn_outputs)
        return ffn_outputs, self_attn_probs, attn_probs


class Decoders(nn.Module):
    def __init__(self, config):
        super(Decoders, self).__init__()
        self.padding_idx = config.padding_idx
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, enc_ids, enc_outputs, dec_ids, dec_embeddings):
        # dec_self_pad_mask : [batch_size, dec_seq_length, dec_seq_length]
        dec_self_pad_mask = get_pad_mask(dec_ids, dec_ids, self.padding_idx)
        # look_ahead_mask : [batch_size, dec_seq_length, dec_seq_length]
        look_ahead_mask = get_look_ahead_mask(dec_ids)
        look_ahead_mask = torch.gt((look_ahead_mask + dec_self_pad_mask), 0)

        # pad_mask : [batch_size, dec_seq_length, enc_seq_length]
        pad_mask = get_pad_mask(dec_ids, enc_ids, self.padding_idx)

        # hidden_states: [batch_size, dec_seq_length, hidden_size]
        hidden_states = dec_embeddings
        self_attn_probs, enc_dec_attn_probs = [], []
        for i, layer in enumerate(self.layers):
            hidden_states, self_attn_prob, attn_prob = layer(hidden_states, enc_outputs, look_ahead_mask, pad_mask)
            self_attn_probs.append(self_attn_prob)
            enc_dec_attn_probs.append(attn_prob)
        return hidden_states, self_attn_probs, enc_dec_attn_probs


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.padding_idx = config.padding_idx
        self.share_embeddings = config.share_embeddings

        self.embedding = Embedding(config)
        self.encoders = Encoders(config)
        self.decoders = Decoders(config)
        self.dense = nn.Linear(config.hidden_size, config.trg_vocab_size)

    def forward(self, enc_ids, dec_ids):
        # enc_embeddings : [batch_size, enc_seq_length, hidden_size]
        # dec_embeddings : [batch_size, dec_seq_length, hidden_size]
        enc_embeddings, dec_embeddings = self.embedding(enc_ids, dec_ids)

        # enc_outputs : [batch_size, enc_seq_length, hidden_size]
        enc_outputs, enc_attn_probs = self.encoders(enc_ids, enc_embeddings)

        # dec_outputs : [batch_size, dec_seq_length, hidden_size]
        dec_outputs, masked_attn_probs, dec_attn_probs = self.decoders(enc_ids, enc_outputs, dec_ids, dec_embeddings)

        total_attn_probs = dict()
        total_attn_probs['enc_attn_probs'] = enc_attn_probs
        total_attn_probs['masked_attn_probs'] = masked_attn_probs
        total_attn_probs['dec_attn_probs'] = dec_attn_probs

        # outputs : [batch_size, dec_seq_length, trg_vocab_size]
        outputs = self.dense(dec_outputs)
        return outputs, total_attn_probs


if __name__ == '__main__':
    config = TransformerConfig(src_vocab_size=5000,
                               trg_vocab_size=5000,
                               hidden_size=256,
                               num_attn_head=4,
                               share_embeddings=False,
                               feed_forward_size=1024,
                               enc_max_seq_length=512,
                               dec_max_seq_length=128)
    inputs = torch.randint(5000, (2, 16), dtype=torch.long, device=config.device)

    model = Transformer(config).to(config.device)
    outputs, attns = model(inputs, inputs)
    print('model output:', outputs.size())
