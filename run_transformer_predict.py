import torch
from model.transformer import Transformer, TransformerConfig
from utils import text2ids
from transformers import ElectraTokenizer


def predict(config, tokenizer, model, text):
    if type(text) is str:
        texts = [text]
    sos = tokenizer.convert_tokens_to_ids(['[CLS]'])  # use cls token as start of sequence
    eos = tokenizer.convert_tokens_to_ids(['[SEP]'])  # use sep token as end of sequence

    enc_ids = text2ids(texts, tokenizer, config, 'encoder')
    dec_ids = torch.zeros(len(enc_ids), config.dec_max_seq_length, dtype=torch.long, device=config.device)

    next_idx = sos[0]
    for i in range(config.dec_max_seq_length):
        if next_idx == eos[0]:
            dec_ids = dec_ids[:, 1:i][0]  # delete start, end token
            break
        dec_ids[0][i] = next_idx
        output, _ = model(enc_ids, dec_ids)
        output = output.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_idx = output[i].item()
    ids = dec_ids.to('cpu').numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    text = tokenizer.convert_tokens_to_string(tokens)
    return text


if __name__ == '__main__':
    # we use pretrained tokenizer from monologg github
    tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    vocab = tokenizer.get_vocab()
    src_vocab_size = len(vocab)
    trg_vocab_size = len(vocab)

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

    model_path = './weight/transformer_50'
    model.load_state_dict(torch.load(model_path))

    sentences = ['책 한 권을 빌리시게 되면 지금으로부터 사주 동안 빌릴 수 있습니다. 할인해주세요.',
                 '이벤트 할인은 일일 일회 제한이며 십퍼센트 할인이 가능하며 중복 할인은 적용되지 않습니다.',
                 '해당 상품은 만이천팔백원입니다.',
                 '번호는 공일공 다시 구구공공 다시 공구팔구이고 이전에 두번 방문했습니다.',
                 '고객님의 객실은 비동 천삼백이호이고 객실키는 2개 제공됩니다.']
    for s in sentences:
        result = predict(config, tokenizer, model, sentences)
        print('predict reuslt:', result)
