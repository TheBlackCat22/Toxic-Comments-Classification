import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import utils

class Toxic_Comment_Dataset(Dataset):
    def __init__(self, X_df, y_df):
        print("Preprocessing:")
        X_df.insert(loc=2, column='cleaned', value=X_df['comment_text'].apply(lambda x: utils.text_cleaner(x)))
        X_df = utils.text_preprocessor(X_df)
        self.X_df = X_df['lemmatized_without_stopwords'].values
        self.y_df = y_df.iloc[:, :-1].values

    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, index):
        label = list(map(float, self.y_df[index]))
        text = str(self.X_df[index])
        return (label, text)


def get_vocabulary(train_dataset):
    train_iter = iter(train_dataset)
    tokenizer = get_tokenizer('basic_english')
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"], min_freq=2)
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def get_dataloader(train_dataset, val_dataset, vocab, batch_size):
    tokenizer = get_tokenizer('basic_english')
    text_pipeline = lambda x: vocab(tokenizer(x))

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
             label_list.append(_label)
             processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
             text_list.append(processed_text)
             offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list).float()
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list, text_list, offsets

    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size= len(val_dataset), shuffle=True, collate_fn=collate_batch)

    return train_loader, val_loader