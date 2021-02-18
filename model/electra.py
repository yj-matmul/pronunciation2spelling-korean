import numpy as np
import math
import torch
import torch.nn as nn


'''
    this config is referenced huggingface ElectraConfig
'''

class ElectraConfig:
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

electra_config = ElectraConfig(vocab_size=35000,
                                       embedding_size=768,
                                       hidden_size=768,
                                       intermediate_size=3072,
                                       max_position_embeddings=512,
                                       num_attention_heads=12)