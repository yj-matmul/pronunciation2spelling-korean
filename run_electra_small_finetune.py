import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from model.transformer import TransformerConfig, Decoders
from utils import text2ids
from transformers import ElectraTokenizer, ElectraModel, ElectraConfig
import glob
import time


class CustomDataset(Dataset):
    def __init__(self, src_lines, trg_lines, tokenizer, config):
        self.enc_input = text2ids(src_lines, tokenizer, config, 'encoder')
        self.dec_input = text2ids(trg_lines, tokenizer, config, 'decoder')
        self.target = text2ids(trg_lines, tokenizer, config, 'target')

    def __len__(self):
        return len(self.enc_input)

    def __getitem__(self, idx):
        x = self.enc_input[idx]
        y = self.dec_input[idx]
        z = self.target[idx]
        return x, y, z


class Pronunciation2Spelling(nn.Module):
    def __init__(self, enc_config, dec_config, first_train):
        super(Pronunciation2Spelling, self).__init__()
        # KoELECTRA-base-v3
        self.electra = ElectraModel(enc_config)
        if first_train:
            self.electra.load_state_dict(torch.load('../pretrained/electra_pretrained_small'))
        self.embedding = self.electra.get_input_embeddings()
        if enc_config.embedding_size != dec_config.hidden_size:
            self.embedding_projection = nn.Linear(enc_config.embedding_size, dec_config.hidden_size)
        self.decoders = Decoders(dec_config)
        self.dense = nn.Linear(dec_config.hidden_size, dec_config.trg_vocab_size)

        self.padding_idx = dec_config.padding_idx

    def forward(self, enc_ids, dec_ids):
        dec_embeddings = self.embedding(dec_ids)
        if hasattr(self, 'embedding_projection'):
            dec_embeddings = self.embedding_projection(dec_embeddings)
        enc_outputs = self.electra(enc_ids).last_hidden_state
        dec_outputs, _, _ = self.decoders(enc_ids, enc_outputs, dec_ids, dec_embeddings)
        model_output = self.dense(dec_outputs)
        return model_output


if __name__ == '__main__':
    # we use pretrained tokenizer, encoders from monologg github
    tokenizer = ElectraTokenizer.from_pretrained('../pretrained/huggingface_korean')
    vocab = tokenizer.get_vocab()
    electra_vocab_size, decoder_src_vocab_size, decoder_trg_vocab_size = len(vocab), len(vocab), len(vocab)

    # please set your custom data path
    src_file_path = 'D:/Storage/side_project_data/pronunciation2spelling-korean/pronunciation.txt'
    trg_file_path = 'D:/Storage/side_project_data/pronunciation2spelling-korean/spelling.txt'
    with open(src_file_path, 'r', encoding='utf8') as f:
        src_lines = list(map(lambda x: x.strip('\n'), f.readlines()))
    with open(trg_file_path, 'r', encoding='utf8') as f:
        trg_lines = list(map(lambda x: x.strip('\n'), f.readlines()))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # electra config values are fixed since this model is already pretrained
    electra_config = ElectraConfig(vocab_size=35000,
                                   embedding_size=128,
                                   hidden_size=256,
                                   intermediate_size=1024,
                                   max_position_embeddings=512,
                                   num_attention_heads=4)
    decoder_config = TransformerConfig(src_vocab_size=decoder_src_vocab_size,
                                       trg_vocab_size=decoder_trg_vocab_size,
                                       hidden_size=256,
                                       num_hidden_layers=12,
                                       num_attn_head=4,
                                       hidden_act='gelu',
                                       device=device,
                                       feed_forward_size=1024,
                                       padding_idx=0,
                                       share_embeddings=True,
                                       enc_max_seq_length=128,
                                       dec_max_seq_length=128)
    dataset = CustomDataset(src_lines, trg_lines, tokenizer, decoder_config)

    first_train = False
    batch_size = 32
    lr = 1e-4
    dataset = CustomDataset(src_lines, trg_lines, tokenizer, decoder_config)
    total_iteration = int(dataset.__len__() // batch_size) + 1
    log_iteration = total_iteration // 2
    patience = 1  # total_iteration // 2
    plus_epoch = 10

    model = Pronunciation2Spelling(electra_config, decoder_config, first_train).to(decoder_config.device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(ignore_index=decoder_config.padding_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=patience)

    if first_train:
        # first finetune (only have encoder pretrained weight)
        last_epoch = 0
        total_epoch = plus_epoch

    else:
        # continue finetune
        weights = glob.glob('./weight/electra_small_*')
        last_epoch = int(weights[-1].split('_')[-1])
        weight_path = weights[-1].replace('\\', '/')
        print('weight info of last epoch', weight_path)
        model.load_state_dict(torch.load(weight_path))
        total_epoch = last_epoch + plus_epoch


    model.train()
    start_time = time.time()
    for epoch in range(plus_epoch):
        epoch_loss = 0
        for iteration, data in enumerate(data_loader):
            encoder_inputs, decoder_inputs, targets = data
            optimizer.zero_grad()
            logits = model(encoder_inputs, decoder_inputs)
            logits = logits.contiguous().view(-1, decoder_trg_vocab_size)
            targets = targets.contiguous().view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss
            if (iteration + 1) % log_iteration == 0:
                print('Epoch: %3d\t' % (last_epoch + epoch + 1),
                      'Iteration: %4d\t' % (iteration + 1),
                      'Cost: {:.5f}\t'.format(epoch_loss/(iteration + 1)),
                      'LR: {:.6f}'.format(optimizer.param_groups[0]['lr']))
        scheduler.step(epoch_loss)
    print('running time for train: {:.2f}'.format((time.time() - start_time) / 60))
    model_path = './weight/electra_small_%d' % total_epoch
    torch.save(model.state_dict(), model_path)
