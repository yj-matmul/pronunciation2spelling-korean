import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from model.transformer import Transformer, TransformerConfig
from utils import text2ids
from transformers import ElectraTokenizer
import glob


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


if __name__ == '__main__':
    # we use pretrained tokenizer from monologg github
    tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    vocab = tokenizer.get_vocab()
    src_vocab_size = len(vocab)
    trg_vocab_size = len(vocab)

    # load your data
    src_file_path = 'D:/Storage/side_project_data/pronunciation2spelling-korean/pronunciation.txt'
    trg_file_path = 'D:/Storage/side_project_data/pronunciation2spelling-korean/spelling.txt'
    with open(src_file_path, 'r', encoding='utf8') as f:
        src_lines = list(map(lambda x: x.strip('\n'), f.readlines()))
    with open(trg_file_path, 'r', encoding='utf8') as f:
        trg_lines = list(map(lambda x: x.strip('\n'), f.readlines()))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = TransformerConfig(src_vocab_size=src_vocab_size,
                               trg_vocab_size=trg_vocab_size,
                               hidden_size=512,
                               num_hidden_layers=6,
                               num_attn_head=8,
                               hidden_act='gelu',
                               device=device,
                               feed_forward_size=2048,
                               padding_idx=0,
                               share_embeddings=True,
                               enc_max_seq_length=128,
                               dec_max_seq_length=128)

    model = Transformer(config).to(config.device)

    dataset = CustomDataset(src_lines, trg_lines, tokenizer, config)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=2)

    train_continue = False
    plus_epoch = 100
    if train_continue:
        weights = glob.glob('./weight/transformer_*')
        last_epoch = int(weights[-1].split('_')[-1])
        weight_path = weights[-1].replace('\\', '/')
        print('weight info of last epoch', weight_path)
        model.load_state_dict(torch.load(weight_path))
        total_epoch = last_epoch + plus_epoch
    else:
        last_epoch = 0
        total_epoch = plus_epoch

    model.train()
    for epoch in range(plus_epoch):
        epoch_loss = 0
        for iteration, data in enumerate(data_loader):
            encoder_inputs, decoder_inputs, targets = data
            optimizer.zero_grad()
            logits, _ = model(encoder_inputs, decoder_inputs)
            logits = logits.contiguous().view(-1, trg_vocab_size)
            targets = targets.contiguous().view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss
            if (iteration + 1) % 200 == 0:
                print('Epoch: %3d\t' % (last_epoch + epoch + 1),
                      'Iteration: %3d \t' % (iteration + 1),
                      'Cost: {:.5f}'.format(epoch_loss/(iteration + 1)))
        scheduler.step(epoch_loss)
    model_path = './weight/transformer_%d' % total_epoch
    torch.save(model.state_dict(), model_path)
