import re
import random
import pandas as pd

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

FILEPATH = "yelp_datasets/"
PITT_MEX_REVIEW_FILENAME = "pittsburgh_mexican_yelp_reviews.csv"
SENTIMENT_FEATURES_CSV = "sentiment_features.csv"


def get_clean_words(text):
    all_words = []

    # lower text
    text = text.lower()

    # remove punctuations
    cleaned = re.sub(r"[^(a-zA-Z)\s]", "", text)

    # tokenize
    tokenized = word_tokenize(cleaned)

    # remove words that contain numbers
    text = [word for word in tokenized if not any(c.isdigit() for c in word)]

    # remove stopwords
    stop_words = set(stopwords.words("english"))
    stopped = [w for w in text if not w in stop_words]

    # remove empty tokens
    non_empty = [t for t in stopped if len(t) > 0]

    # parts of speech tagging for each word
    pos_tags = nltk.pos_tag(non_empty)

    # lemmatize text
    text = [
        WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags
    ]

    # remove words with only one letter
    text = [t for t in text if len(t) > 1]

    # join all
    text = " ".join(text)

    return text


def get_wordnet_pos(pos_tag):

    if pos_tag.startswith("J"):
        return wordnet.ADJ

    elif pos_tag.startswith("V"):
        return wordnet.VERB

    elif pos_tag.startswith("N"):
        return wordnet.NOUN

    elif pos_tag.startswith("R"):
        return wordnet.ADV

    else:
        return wordnet.NOUN


def build_features(filepath, pitt_mex_review_filename):
    reviews = pd.read_csv(filepath + pitt_mex_review_filename)
    reviews["is_good_review"] = reviews["stars"].apply(lambda x: 1 if x >= 4 else 0)
    reviews["is_good_review"].value_counts()

    stop_words = list(set(stopwords.words("english")))
    reviews["words"] = reviews["text"].apply(lambda x: get_clean_words(x))

    # now we'll add features to build a predictive model for positive reviews
    # add sentiment intensity scores for reviews
    sid = SentimentIntensityAnalyzer()
    reviews["sentiments"] = reviews["text"].apply(lambda x: sid.polarity_scores(x))
    reviews = pd.concat(
        [reviews.drop(["sentiments"], axis=1), reviews["sentiments"].apply(pd.Series)],
        axis=1,
    )

    # add number of characters column
    reviews["num_chars"] = reviews["text"].apply(lambda x: len(x))

    # add number of words column
    reviews["num_words"] = reviews["text"].apply(lambda x: len(x.split(" ")))

    documents = [
        TaggedDocument(doc, [i])
        for i, doc in enumerate(reviews["words"].apply(lambda x: x.split(" ")))
    ]

    # train a Doc2Vec model with our text data
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

    # transform each document into a vector data
    doc2vec_df = (
        reviews["words"]
        .apply(lambda x: model.infer_vector(x.split(" ")))
        .apply(pd.Series)
    )
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    reviews = pd.concat([reviews, doc2vec_df], axis=1)

    # create word vector features
    tfidf = TfidfVectorizer(min_df=10)
    tfidf_result = tfidf.fit_transform(reviews["words"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = reviews.index
    reviews = pd.concat([reviews, tfidf_df], axis=1)

    label = "is_good_review"
    ignore_cols = [
        label,
        "business_id",
        "cool",
        "date",
        "funny",
        "review_id",
        "stars",
        "text",
        "words",
        "useful",
        "user_id",
    ]
    features = [c for c in reviews.columns if c not in ignore_cols]

    # create a test,train split
    X_train, X_test, y_train, y_test = train_test_split(
        reviews[features], reviews[label], test_size=0.20, random_state=42
    )

    return X_train, X_test, y_train, y_test, features


def build_lrc_model(X_train, X_test, y_train, y_test, features, sentiment_feature_csv):
    # build a linear regression model to predict positive reviews
    LRC = linear_model.LogisticRegression()
    LRC.fit(X_train, y_train)

    predicted = LRC.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, predicted))
    print("Confusion matrix", metrics.confusion_matrix(predicted, y_test))

    # save list of features and coefficients
    pd.DataFrame(zip(LRC.coef_[0], features)).sort_values(0).rename(
        columns={0: "LCF_coef", 1: "feature"}
    ).to_csv(sentiment_features_csv)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = build_features(
        FILEPATH, PITT_MEX_REVIEW_FILENAME
    )
    build_lrc_model(X_train, X_test, y_train, y_test, features, SENTIMENT_FEATURES_CSV)
