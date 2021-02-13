import re
import torch


def text_normalization(text):
    text = text.strip()
    # print('raw:', text)
    text = re.sub(r'([?.,!+-])', r' \1 ', text)
    # print('특수기호:', text)
    text = re.sub(r'[" "]+', " ", text)

    text = re.sub(r'[^0-9a-zA-Z가-힣/?.,~!@#$%&*()+_-]+', ' ', text)
    text = text.strip()
    # print('최종:', text)
    return text


def text2ids(text_list, tokenizer, config, mode=str, use_test_normalization=False):
    if type(mode) is not str:
        print('must choose mode in [encoder, decoder, target]')

    pad = tokenizer.convert_tokens_to_ids(['[PAD]'])  # use pad token
    sos = tokenizer.convert_tokens_to_ids(['[CLS]'])  # use cls token as start of sequence
    eos = tokenizer.convert_tokens_to_ids(['[SEP]'])  # use sep token as end of sequence

    max_length = 0  # notify max_seq_length you need to set
    ids_list = []
    for text in text_list:
        if use_test_normalization:
            text = text_normalization(text)
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        max_length = max(max_length, len(ids) + 1)  # plus one for special token
        if mode == 'encoder':
            ids += pad * (config.enc_max_seq_length - len(ids))
        elif mode == 'decoder':
            ids = sos + ids
            ids += pad * (config.dec_max_seq_length - len(ids))
        else:
            ids += eos
            ids += pad * (config.dec_max_seq_length - len(ids))
        ids_list.append(ids)
    if config.dec_max_seq_length < max_length:
        print('you need to set seq length more than', max_length)
    ids_list = torch.LongTensor(ids_list).to(config.device)
    return ids_list


if __name__ == '__main__':
    sample = '확인해 드릴게요, 세금을 포함해서 102만 원이라고 나오네요.'
    print(text_normalization(sample))
