import csv
import json
import pandas as pd
import contractions
import re
import string
import argparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.corpus import brown, wordnet
from nltk.stem import WordNetLemmatizer

def get_args():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--train_data_path', type=str, default="../data/training_data.csv")
    parser.add_argument('--test_data_path', type=str, default="../data/test_data.csv")
    parser.add_argument('--net', type=str, default='FF', choices=['FF', 'RNN', 'LSTM', 'Trans'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default= 5)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--save_folder', type=str, default="./logs/")
    args = parser.parse_args()
    return args


def data_importer(path, train):
    if train:
        with open(path, "r") as f:
            csvreader = csv.reader(f, delimiter="\n")
            columns = next(csvreader)[0].split(",")
            data = []
            for row in csvreader:
                sample = row[0]
                comma_idx = [i for i, ltr in enumerate(sample) if ltr == ',']
                temp_list = [sample[:(comma_idx[0])], sample[(comma_idx[0]+1):comma_idx[-7]], sample[(comma_idx[-7]+1):comma_idx[-6]], sample[(comma_idx[-6]+1):comma_idx[-5]], sample[(
                    comma_idx[-5]+1):comma_idx[-4]], sample[(comma_idx[-4]+1):comma_idx[-3]], sample[(comma_idx[-3]+1):comma_idx[-2]], sample[(comma_idx[-2]+1):comma_idx[-1]], sample[(comma_idx[-1]+1):]]
                data.append(temp_list)
        training_data = pd.DataFrame(data, columns=columns)

        print(f"Shape of Training Data is {training_data.shape}")
        print(f'Columns in Training Data: {training_data.columns.values}')

        return training_data
    else:
        with open(path, "r") as f:
            csvreader = csv.reader(f, delimiter="\n")
            columns = next(csvreader)[0].split(",")
            data = []
            for row in csvreader:
                sample = row[0]
                comma_idx = [i for i, ltr in enumerate(sample) if ltr == ',']
                temp_list = [sample[:(comma_idx[0])], sample[(comma_idx[0]+1):]]
                data.append(temp_list)
        test_data = pd.DataFrame(data, columns=columns)

        print(f"Shape of Testing Data is {test_data.shape}")
        print(f'Columns in Testing Data: {test_data.columns.values}')
        
        return test_data


def text_cleaner(text):
    # lower Case
    def make_lower(text):
        return text.lower()

    # expand contractions
    def expand_contractions(text):
        return contractions.fix(text)

    # remove URLs
    def remove_url(text):
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    # remove HTML tags
    def remove_html(text):
        html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
        return re.sub(html, "", text)

    # remove Non-ASCI
    def remove_non_ascii(text):
        return re.sub(r'[^\x00-\x7f]', r'', text)

    # remove special charecters
    def remove_special_characters(text):
        emoji_pattern = re.compile(
            '['
            u'\U0001F600-\U0001F64F'  # emoticons
            u'\U0001F300-\U0001F5FF'  # symbols & pictographs
            u'\U0001F680-\U0001F6FF'  # transport & map symbols
            u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
            u'\U00002702-\U000027B0'
            u'\U000024C2-\U0001F251'
            ']+',
            flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    # remove punctuations
    def remove_punct(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    # remove numbers
    def remove_num(text):
        return text.translate(str.maketrans('', '', string.digits))

    text = make_lower(text)
    text = expand_contractions(text)
    text = remove_url(text)
    text = remove_html(text)
    text = remove_non_ascii(text)
    text = remove_special_characters(text)
    text = remove_punct(text)
    text = remove_num(text)
    return text


def text_preprocessor(data):
    tokenized = data['cleaned'].apply(word_tokenize)

    # Removing Stopwords
    stop = set(stopwords.words('english'))
    stopwords_removed = tokenized.apply(
        lambda x: [word for word in x if word not in stop])
    print("  -Removed Stopwords")

    # POS Tagging
    wordnet_map = {"N": wordnet.NOUN,
                   "V": wordnet.VERB,
                   "J": wordnet.ADJ,
                   "R": wordnet.ADV
                   }

    train_sents = brown.tagged_sents(categories='news')
    t0 = nltk.DefaultTagger('NN')
    t1 = nltk.UnigramTagger(train_sents, backoff=t0)
    t2 = nltk.BigramTagger(train_sents, backoff=t1)

    def pos_tag_wordnet(text):
        pos_tagged_text = t2.tag(text)
        pos_tagged_text = [(word, wordnet_map.get(pos_tag[0])) if pos_tag[0] in wordnet_map.keys(
        ) else (word, wordnet.NOUN) for (word, pos_tag) in pos_tagged_text]
        return pos_tagged_text

    pos_tagged_tokens_without_stopwords = stopwords_removed.apply(
        lambda x: pos_tag_wordnet(x))
    del wordnet_map, train_sents, t0, t1, t2, pos_tag_wordnet
    print("  -Created PosTags")

    # Lemmatization
    def lemmatize_word(text):
        lemmatizer = WordNetLemmatizer()
        lemma = [lemmatizer.lemmatize(word, tag) for word, tag in text]
        return lemma

    lemmatize_word_w_pos = pos_tagged_tokens_without_stopwords.apply(
        lambda x: lemmatize_word(x))
    lemmatize_word_w_pos = lemmatize_word_w_pos.apply(
        lambda x: [word for word in x if word not in stop])  # double check to remove stop words
    lemmatized_without_stopwords = [
        ' '.join(map(str, l)) for l in lemmatize_word_w_pos]  # join back to text
    data.insert(loc=3, column='lemmatized_without_stopwords',
                value=lemmatized_without_stopwords)
    print("  -Lemmatized")

    return data


def pretty_json(hp):
  json_hp = json.dumps(hp, indent=2)
  return "".join("\t" + line for line in json_hp.splitlines(True))