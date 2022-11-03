# Preprocessing
def text_preprocessor(data):
    # Tokenization
    print("-Completed Tokenization")
    from nltk.tokenize import word_tokenize

    tokenized = data['cleaned'].apply(word_tokenize)

    # Removing Stopwords
    from nltk.corpus import stopwords

    stop = set(stopwords.words('english'))
    stopwords_removed = tokenized.apply(
        lambda x: [word for word in x if word not in stop])
    data.insert(loc=3, column='stopwords_removed',
                value=stopwords_removed.apply(lambda x: ' '.join(x)))
    print("-Removed Stopwords")

    # POS Tagging
    import nltk
    from nltk.corpus import brown, wordnet

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
    pos_tagged_tokens_with_stopwords = tokenized.apply(
        lambda x: pos_tag_wordnet(x))
    del wordnet_map, train_sents, t0, t1, t2, pos_tag_wordnet
    print("-Created PosTags")

    # Lemmatization
    from nltk.stem import WordNetLemmatizer

    def lemmatize_word(text):
        lemmatizer = WordNetLemmatizer()
        lemma = [lemmatizer.lemmatize(word, tag) for word, tag in text]
        return lemma

    # with stop words
    lemmatize_word_w_pos = pos_tagged_tokens_with_stopwords.apply(
        lambda x: lemmatize_word(x))
    lemmatized_with_stopwords = [
        ' '.join(map(str, l)) for l in lemmatize_word_w_pos]  # join back to text
    data.insert(loc=4, column='lemmatized_with_stopwords',
                value=lemmatized_with_stopwords)
    del lemmatized_with_stopwords, lemmatize_word_w_pos

    # without stop words
    lemmatize_word_w_pos = pos_tagged_tokens_without_stopwords.apply(
        lambda x: lemmatize_word(x))
    lemmatize_word_w_pos = lemmatize_word_w_pos.apply(
        lambda x: [word for word in x if word not in stop])  # double check to remove stop words
    lemmatized_without_stopwords = [
        ' '.join(map(str, l)) for l in lemmatize_word_w_pos]  # join back to text
    data.insert(loc=5, column='lemmatized_without_stopwords',
                value=lemmatized_without_stopwords)
    del lemmatized_without_stopwords, lemmatize_word_w_pos, lemmatize_word
    print("-Completed Lemmatization")

    # Stemming
    import sys
    sys.setrecursionlimit(20000)

    def porter_stemmer(text):
        stemmer = nltk.PorterStemmer()
        stems = [stemmer.stem(i) for i in text]
        return stems

    # with stop words
    stemmed_with_stopwords = tokenized.apply(lambda x: porter_stemmer(x))
    stemmed_with_stopwords = stemmed_with_stopwords.apply(
        lambda x: ' '.join(x))
    data.insert(loc=6, column='stemmed_with_stopwords',
                value=stemmed_with_stopwords)
    del stemmed_with_stopwords

    # without stop words
    stemmed_without_stopwords = stopwords_removed.apply(
        lambda x: porter_stemmer(x))
    stemmed_without_stopwords = stemmed_without_stopwords.apply(
        lambda x: ' '.join(x))
    data.insert(loc=7, column='stemmed_without_stopwords',
                value=stemmed_without_stopwords)
    del stemmed_without_stopwords, porter_stemmer
    print("-Completed Stemming")

    return data
