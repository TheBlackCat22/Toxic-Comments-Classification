import warnings
from sklearn.metrics import classification_report, multilabel_confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.filterwarnings("ignore")


def get_best_classifier(X_train, y_train, for_doc2vec=False):
    parameters = [
        {
            'clf': LogisticRegression(class_weight='balanced', max_iter=10000),
            'clf__solver': ['newton-cg', 'lbfgs', 'liblinear']
        },
        {
            'clf': MultinomialNB(fit_prior=True, class_prior=None),
            'clf__alpha': [10**-9, 10**-4, 1, 5]
        },
        {
            'clf': KNeighborsClassifier(n_jobs=-1),
            'clf__n_neighbors': list(range(1, 18, 4))
        },
        {
            'clf': DecisionTreeClassifier(),
            'clf__criterion': ('gini', 'entropy'),
            'clf__max_features': ('sqrt', 'log2'),
            'clf__ccp_alpha': (0.01, 0.03, 0.05, 0.07, 0.1)
        },
        {
            'clf': svm.SVC(),
            'clf__C': [0.1, 1]
        }
    ]
    if for_doc2vec:
        parameters.pop(1)
        parameters.pop(-1)

    print(parameters)

    max_f1 = 0
    for params in parameters:
        clf = params.pop('clf')
        print(f"        -{str(clf)}")

        steps = [
            ("scalar", StandardScaler(with_mean=False)),
            ('feature_selector', SelectKBest(chi2, k=1000)),
            ('clf', clf)
        ]

        if for_doc2vec:
            steps.pop(1)

        pipe = Pipeline(steps)
        grid_model = GridSearchCV(pipe, param_grid=params, scoring='f1_micro',
                                  cv=10, n_jobs=-1)
        grid_model.fit(X_train, y_train)

        print("          -f1_micro Score: ", grid_model.best_score_)
        if grid_model.best_score_ > max_f1:
            result = {
                'clf': str(clf),
                'best_score': grid_model.best_score_,
                'best_params': grid_model.best_estimator_,
                'grid': grid_model
            }
    return result


def train_classifiers(vector_dict, y_train, model_save_path):
    # Bag of Words
    print("\n  -Bag of Words")
    vector_dict_bow = vector_dict["bow"]
    results_bow = {}
    for preprocessing_type in vector_dict_bow.keys():
        print(f"    -{preprocessing_type}")
        [train_vectors, _, _] = vector_dict_bow[preprocessing_type]
        results_pp_type = {}
        for label in y_train.columns:
            print(f"      -{label}")
            result = get_best_classifier(
                train_vectors, y_train[label].values)
            results_pp_type[label] = result
            print("        -Best f1_micro score: ", result["best_score"])
        results_bow[preprocessing_type] = results_pp_type
    # Saving Results
    with open(model_save_path+'_bow.pkl', "wb") as f:
        pickle.dump(results_bow, f)
    print(f"  -Results Saved to {model_save_path+'_bow.pkl'}")

    # Tfidf
    print("\n  -Tfidf")
    vector_dict_tfidf = vector_dict["tfidf"]
    results_tfidf = {}
    for preprocessing_type in vector_dict_tfidf.keys():
        print(f"    -{preprocessing_type}")
        [train_vectors, _, _] = vector_dict_tfidf[preprocessing_type]
        results_pp_type = {}
        for label in y_train.columns:
            print(f"      -{label}")
            result = get_best_classifier(
                train_vectors, y_train[label].values)
            results_pp_type[label] = result
            print("        -Best f1_micro score: ", result["best_score"])
        results_tfidf[preprocessing_type] = results_pp_type
    # Saving Results
    with open(model_save_path+'_tfidf.pkl', "wb") as f:
        pickle.dump(results_tfidf, f)
    print(f"  -Results Saved to {model_save_path+'_tfidf.pkl'}")

    # Doc2Vec
    print("\n  -Doc2Vec")
    vector_dict_doc2vec = vector_dict["doc2vec"]
    results_doc2vec = {}
    for preprocessing_type in vector_dict_doc2vec.keys():
        print(f"    -{preprocessing_type}")
        [train_vectors, _, _] = vector_dict_doc2vec[preprocessing_type]
        results_pp_type = {}
        for label in y_train.columns:
            print(f"      -{label}")
            result = get_best_classifier(
                train_vectors, y_train[label].values, for_doc2vec=True)
            results_pp_type[label] = result
            print("        -Best f1_micro score: ", result["best_score"])
        results_doc2vec[preprocessing_type] = results_pp_type
    # Saving Results
    with open(model_save_path+'_doc2vec.pkl', "wb") as f:
        pickle.dump(results_doc2vec, f)
    print(f"Results Saved to {model_save_path+'_doc2vec.pkl'}")


def get_predictions(X_val, model_dict):
    labels = model_dict.keys()
    predictions = []
    for label in labels:
        grid = model_dict[label]['grid']
        pred = np.array(grid.predict(X_val))
        predictions.append(pred)
    return np.array(predictions).T


def validate_classifiers(y_val, vector_dict, results_dict):
    preprocessing_types = list(results_dict.keys())
    labels = list(results_dict[preprocessing_types[0]].keys())
    for preprocessing_type in preprocessing_types:
        model_dict = results_dict[preprocessing_type]
        _, X_val, _ = vector_dict[preprocessing_type]
        predictions = get_predictions(X_val, model_dict)
        print(f'\n\n    -{preprocessing_type}:')
        print(classification_report(y_val, predictions,
              output_dict=False, target_names=labels, zero_division=0))
        cfms = multilabel_confusion_matrix(y_val, predictions)
        for i, label in enumerate(labels):
            print("\n", label)
            print(cfms[i])
        print(
            f"\nMacro ROC_AUC Score = {roc_auc_score(y_val, predictions, average='macro')}")
