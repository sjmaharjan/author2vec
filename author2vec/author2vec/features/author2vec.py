from gensim.models import Doc2Vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from author_style.author_style_emb import transform_doc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
import os
from sklearn.preprocessing import normalize

from functools import partial

__all__ = [
    "AuthorFeatures",
    "AuthorInferVecFeatures",
    "Author2VecFeatures",
    "TfidfEmbeddingVectorizer",
    "MeanEmbeddingVectorizer",
]


class AuthorFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_features_ = 519

    def get_feature_names(self):
        return np.array(
            ["Author_{}".format(author) for author in range(self.num_features_)]
        )

    def make_feature_vec(self, author):
        feature_vec = np.zeros((self.num_features_,))
        author_id = int(author.split("_")[-1])
        feature_vec[author_id] = 1
        return feature_vec

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        doc_feature_vecs = np.zeros((len(documents), self.num_features_))

        for i, doc in enumerate(documents):
            # Print a status message every  200 doc
            if i % 200 == 0.0:
                print("Document %d of %d" % (i, len(documents)))

            doc_feature_vecs[i] = self.make_feature_vec(doc.author2vec_id)

        return doc_feature_vecs


class Author2VecFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, model_dir=None, model_name=None, norm=None, dtype=np.float32):
        print("Model dir, model name", model_dir, model_name)
        self.model_dir = model_dir
        self.model_name = model_name
        self.dtype = dtype
        self.norm = norm

    def get_feature_names(self):
        return np.array(["author_emb_" + str(i) for i in range(self.num_features_)])

    def make_feature_vec(self, vec_id):
        try:
            feature_vec = self.model_.docvecs[vec_id]
        except KeyError as e:
            print(e)
            print("Could not fine doc vec for {}".format(vec_id))
            feature_vec = np.zeros((self.num_features_,), dtype=self.dtype)

        return feature_vec

    def fit(self, documents, y=None):
        print("Here===> {}".format(self.model_name))
        if self.model_name:
            print("Loading Vectors")
            model_data = os.path.join(
                self.model_dir, "model_%s.doc2vec" % self.model_name
            )
            if self.model_name == "dbow_dmm" or self.model_name == "dbow_dmc":
                m1 = os.path.join(
                    self.model_dir, "model_%s.doc2vec" % self.model_name.split("_")[0]
                )
                m2 = os.path.join(
                    self.model_dir, "model_%s.doc2vec" % self.model_name.split("_")[1]
                )
                model1 = Doc2Vec.load(m1)
                model2 = Doc2Vec.load(m2)
                self.model_ = ConcatenatedDoc2Vec([model1, model2])
                self.num_features_ = model1.wv.syn0.shape[1] + model2.wv.syn0.shape[1]
            else:
                self.model_ = Doc2Vec.load(model_data)
                self.num_features_ = self.model_.wv.syn0.shape[1]
            print(self.num_features_)
            print("Done Loading vectors")
        else:
            print("Hereeeeee ", self.model_name)
            raise OSError("Model does not exit")

        return self

    def transform(self, documents):
        doc_feature_vecs = np.zeros(
            (len(documents), self.num_features_), dtype=self.dtype
        )

        for i, doc in enumerate(documents):
            # Print a status message every  200 doc
            if i % 200 == 0.0:
                print("Document %d of %d" % (i, len(documents)))
            #
            # print("Document %s" % doc.book_id)
            doc_feature_vecs[i] = self.make_feature_vec(doc.author2vec_id)

        if self.norm:
            print("Vectors normalized")
            doc_feature_vecs = normalize(doc_feature_vecs, norm=self.norm)

        return doc_feature_vecs


class AuthorInferVecFeatures(Author2VecFeatures):
    def __init__(
        self, step, model_dir=None, model_name=None, norm=None, dtype=np.float32
    ):
        super(AuthorInferVecFeatures, self).__init__(
            model_dir=model_dir, model_name=model_name, norm=norm, dtype=dtype
        )
        self.step = step

    def make_feature_vec(self, content):
        n_grams = transform_doc(content, n=3, step=self.step)
        feature_vec = self.model_.infer_vector(n_grams)
        return feature_vec

    def transform(self, documents):
        doc_feature_vecs = np.zeros((len(documents), self.num_features_))

        for i, doc in enumerate(documents):
            # Print a status message every  200 doc
            if i % 200 == 0.0:
                print("Document %d of %d" % (i, len(documents)))

            doc_feature_vecs[i] = self.make_feature_vec(doc.content)

        return doc_feature_vecs


###REF: https://github.com/erogol/QuoraDQBaseline/blob/master/utils.py


class MeanEmbeddingVectorizer(Author2VecFeatures):
    def __init__(
        self, step, model_dir=None, model_name=None, norm=None, dtype=np.float32
    ):
        super(MeanEmbeddingVectorizer, self).__init__(
            model_dir=model_dir, model_name=model_name, norm=norm, dtype=dtype
        )
        self.step = step

    def get_feature_names(self):
        return np.array(["mean_emb_" + str(i) for i in range(self.num_features_)])

    def make_feature_vec(self, content):

        if content:
            return np.mean(
                [
                    self.model_[w]
                    for w in transform_doc(content, n=3, step=self.step)
                    if w in self.model_
                ],
                axis=0,
            )
        else:
            print("Empty content")
            return np.zeros((self.num_features_,), dtype=self.dtype)

    def transform(self, documents):
        doc_feature_vecs = np.zeros(
            (len(documents), self.num_features_), dtype=self.dtype
        )

        for i, doc in enumerate(documents):
            # Print a status message every  200 doc
            if i % 200 == 0.0:
                print("Document %d of %d" % (i, len(documents)))
            #
            # print("Document %s" % doc.book_id)
            doc_feature_vecs[i] = self.make_feature_vec(doc.content)

        if self.norm:
            print("Vectors normalized")
            doc_feature_vecs = normalize(doc_feature_vecs, norm=self.norm)

        return doc_feature_vecs


class TfidfEmbeddingVectorizer(Author2VecFeatures):
    def __init__(
        self, step, model_dir=None, model_name=None, norm=None, dtype=np.float32
    ):
        super(TfidfEmbeddingVectorizer, self).__init__(
            model_dir=model_dir, model_name=model_name, norm=norm, dtype=dtype
        )
        self.step = step
        self.word2weight = None

    def get_feature_names(self):
        return np.array(["tfidf_emb_" + str(i) for i in range(self.num_features_)])

    def fit(self, X, y=None):
        from . import type_ngram_analyze

        super().fit(X, y)
        analyer_fx = partial(type_ngram_analyze, step=self.step)
        tfidf = TfidfVectorizer(analyzer=analyer_fx, lowercase=False)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.max_idf = max_idf
        print("Max idf:{}".format(max_idf))
        self.word2weight = {w: tfidf.idf_[i] for w, i in tfidf.vocabulary_.items()}

        return self

    def make_feature_vec(self, content):
        if content:

            return np.mean(
                [
                    self.model_[w] * self.word2weight.get(w, self.max_idf)
                    for w in transform_doc(content, n=3, step=self.step)
                    if w in self.model_
                ],
                axis=0,
            )
        else:
            print("Empty content")
            return np.zeros((self.num_features_,), dtype=self.dtype)

    def transform(self, documents):
        doc_feature_vecs = np.zeros(
            (len(documents), self.num_features_), dtype=self.dtype
        )

        for i, doc in enumerate(documents):
            # Print a status message every 200 doc
            if i % 200 == 0.0:
                print("Document %d of %d" % (i, len(documents)))
            #
            # print("Document %s" % doc.book_id)
            doc_feature_vecs[i] = self.make_feature_vec(doc.content)

        if self.norm:
            print("Vectors normalized")
            doc_feature_vecs = normalize(doc_feature_vecs, norm=self.norm)

        return doc_feature_vecs
