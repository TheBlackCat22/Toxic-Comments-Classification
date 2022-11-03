import pickle
import numpy as np
from tqdm import tqdm

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def doc2vec_train(data, ln):
    tagged_training_data = [TaggedDocument(words=_d.split(), tags=[str(
        i)]) for i, _d in enumerate(data)]
    # Training
    max_epochs = 100
    trn_model = Doc2Vec(vector_size=ln,
                        alpha=0.025,
                        min_alpha=0.00025,
                        min_count=1,
                        dm=1)
    # Building vocabulary
    trn_model.build_vocab(tagged_training_data)
    # Learning
    for epoch in tqdm(range(max_epochs)):
        trn_model.train(tagged_training_data,
                        total_examples=trn_model.corpus_count,
                        epochs=10)
        # decrease the learning rate
        trn_model.alpha -= 0.0002
        # fix the learning rate, no decay
        trn_model.min_alpha = trn_model.alpha
    return trn_model


def doc2vec_vectorize(X_train, X_val, trn_model):
    trn_vec = []
    for i in range(0, len(X_train)):
        vec = []
        for v in trn_model.dv[i]:
            vec.append(v)
        trn_vec.append(vec)
    X_train = trn_vec
    inf_vec = []
    for row in X_val:
        vec = trn_model.infer_vector(str(row).split())
        inf_vec.append(vec)
    X_val = inf_vec
    return np.array(X_train), np.array(X_val)


def vectorize(X_train, X_val, vector_dict_path):
    vector_dict = {}

    # Bag of Words
    print("\n  -Vecotrizing Using Count Vectorizer")
    vector_dict_bow = {}
    for preprocessing_type in X_train.columns.values[-4:]:
        print(f"    -{preprocessing_type}")
        vectorizer = CountVectorizer(min_df=5, ngram_range=(1, 3))
        temp_train_vectors = vectorizer.fit_transform(
            X_train[preprocessing_type].values.astype(str))
        temp_val_vectors = vectorizer.transform(
            X_val[preprocessing_type].values.astype(str))
        vector_dict_bow[preprocessing_type] = [
            temp_train_vectors, temp_val_vectors, vectorizer]
        print(f"      -Train Vector Shape: {temp_train_vectors.shape}")
        print(f"      -Val Vector Shape: {temp_val_vectors.shape}")
    vector_dict['bow'] = vector_dict_bow

    # Tfidf
    print("\n  -Vectorizing Using Tfidf Vectorizer")
    vector_dict_tfidf = {}
    for preprocessing_type in X_train.columns.values[-4:]:
        print(f"    -{preprocessing_type}")
        vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 3))
        temp_train_vectors = vectorizer.fit_transform(
            X_train[preprocessing_type].values.astype(str))
        temp_val_vectors = vectorizer.transform(
            X_val[preprocessing_type].values.astype(str))
        vector_dict_tfidf[preprocessing_type] = [
            temp_train_vectors, temp_val_vectors, vectorizer]
        print(f"      -Train Vector Shape: {temp_train_vectors.shape}")
        print(f"      -Val Vector Shape: {temp_val_vectors.shape}")
    vector_dict['tfidf'] = vector_dict_tfidf

    # Doc2Vec
    print("\n  -Vectorizing Using Doc2Vec")
    vector_dict_doc2vec = {}
    for preprocessing_type in X_train.columns.values[-4:]:
        print(f"    -{preprocessing_type}")
        print("      -Training")
        trn_model = doc2vec_train(
            X_train[preprocessing_type].values.astype(str), 300)
        print('      -Vectorizing')
        temp_train_vectors, temp_val_vectors = doc2vec_vectorize(X_train[preprocessing_type].values.astype(
            str), X_val[preprocessing_type].values.astype(str), trn_model)
        vector_dict_doc2vec[preprocessing_type] = [
            temp_train_vectors, temp_val_vectors, trn_model]
        print(f"        -Train Vector Shape: {temp_train_vectors.shape}")
        print(f"        -Val Vector Shape: {temp_val_vectors.shape}")
    vector_dict['doc2vec'] = vector_dict_doc2vec

    # Saving Vectors
    with open(vector_dict_path, 'wb') as f:
        pickle.dump(vector_dict, f)
    print(f"Vectors Saved to {vector_dict_path}")
