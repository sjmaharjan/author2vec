from sklearn.pipeline import FeatureUnion
from . import lexical
from . import author2vec
from author2vec.manage import app
from sklearn.feature_extraction.text import TfidfVectorizer
from author2vec.author_style.author_style_emb import transform_doc
from functools import partial
import nltk


__all__ = [
    "lexical",
    "get_feature",
    "create_feature",
    "author2vec",
    "type_ngram_analyze",
]


def preprocess(x):
    return x.replace("\n", " ").replace("\r", "").replace("\x0C", "").lower()


def type_ngram_analyze(x, step, annotate=True):
    return transform_doc(x.content, n=3, step=step, annotate=annotate)


def get_feature(f_name):
    """Factory to create features objects

    Parameters
    ----------
    f_name : features name

    Returns
    ----------
    features: BaseEstimator
        feture object

    """
    features_dic = dict(
        # word ngram
        unigram=lexical.NGramTfidfVectorizer(
            ngram_range=(1, 1),
            preprocessor=preprocess,
            tokenizer=nltk.word_tokenize,
            analyzer="word",
            lowercase=True,
            min_df=2,
        ),
        # char trigram
        char_tri=lexical.NGramTfidfVectorizer(
            ngram_range=(3, 3),
            preprocessor=preprocess,
            analyzer="char",
            lowercase=True,
            min_df=2,
        ),
        # typed character ngrams
        typed_ngram_overlap=TfidfVectorizer(
            analyzer=partial(type_ngram_analyze, step=1), lowercase=False, min_df=2
        ),
        typed_ngram_partial=TfidfVectorizer(
            analyzer=partial(type_ngram_analyze, step=2), lowercase=False, min_df=2
        ),
        typed_ngram_non_overlap=TfidfVectorizer(
            analyzer=partial(type_ngram_analyze, step=3), lowercase=False, min_df=2
        ),
        typed_ngram_overlap_no_annotate=TfidfVectorizer(
            analyzer=partial(type_ngram_analyze, step=1, annotate=False),
            lowercase=False,
            min_df=2,
        ),
        ##########################################################################
        # Mean Embedding Doc Representation
        ##########################################################################
        # book represented as mean of typed ngram embedding
        # dmc
        typed_ngram_overlap_mean_dmc=author2vec.MeanEmbeddingVectorizer(
            step=1, model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dmc"
        ),
        typed_ngram_partial_mean_dmc=author2vec.MeanEmbeddingVectorizer(
            step=2, model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dmc"
        ),
        typed_ngram_non_overlap_mean_dmc=author2vec.MeanEmbeddingVectorizer(
            step=3, model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dmc"
        ),
        # dmm
        typed_ngram_overlap_mean_dmm=author2vec.MeanEmbeddingVectorizer(
            step=1, model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dmm"
        ),
        typed_ngram_partial_mean_dmm=author2vec.MeanEmbeddingVectorizer(
            step=2, model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dmm"
        ),
        typed_ngram_non_overlap_mean_dmm=author2vec.MeanEmbeddingVectorizer(
            step=3, model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dmm"
        ),
        # dbow
        typed_ngram_overlap_mean_dbow=author2vec.MeanEmbeddingVectorizer(
            step=1, model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dbow"
        ),
        typed_ngram_partial_mean_dbow=author2vec.MeanEmbeddingVectorizer(
            step=2, model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dbow"
        ),
        typed_ngram_non_overlap_mean_dbow=author2vec.MeanEmbeddingVectorizer(
            step=3, model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dbow"
        ),
        ##########################################################################
        # Norm book represented as mean of typed ngram embedding
        # dmc
        norm_typed_ngram_overlap_mean_dmc=author2vec.MeanEmbeddingVectorizer(
            step=1, model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dmc", norm="l2"
        ),
        norm_typed_ngram_partial_mean_dmc=author2vec.MeanEmbeddingVectorizer(
            step=2, model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dmc", norm="l2"
        ),
        norm_typed_ngram_non_overlap_mean_dmc=author2vec.MeanEmbeddingVectorizer(
            step=3, model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dmc", norm="l2"
        ),
        # dmm
        norm_typed_ngram_overlap_mean_dmm=author2vec.MeanEmbeddingVectorizer(
            step=1, model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dmm", norm="l2"
        ),
        norm_typed_ngram_partial_mean_dmm=author2vec.MeanEmbeddingVectorizer(
            step=2, model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dmm", norm="l2"
        ),
        norm_typed_ngram_non_overlap_mean_dmm=author2vec.MeanEmbeddingVectorizer(
            step=3, model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dmm", norm="l2"
        ),
        # dbow
        norm_typed_ngram_overlap_mean_dbow=author2vec.MeanEmbeddingVectorizer(
            step=1, model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dbow", norm="l2"
        ),
        norm_typed_ngram_partial_mean_dbow=author2vec.MeanEmbeddingVectorizer(
            step=2, model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dbow", norm="l2"
        ),
        norm_typed_ngram_non_overlap_mean_dbow=author2vec.MeanEmbeddingVectorizer(
            step=3, model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dbow", norm="l2"
        ),
        ##########################################################################
        # TFIDF Embedding Doc Representation
        ##########################################################################
        # book represented as idf*ngram emb of typed ngram embedding
        # dmc
        typed_ngram_overlap_wt_dmc=author2vec.TfidfEmbeddingVectorizer(
            step=1, model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dmc"
        ),
        typed_ngram_partial_wt_dmc=author2vec.TfidfEmbeddingVectorizer(
            step=2, model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dmc"
        ),
        typed_ngram_non_overlap_wt_dmc=author2vec.TfidfEmbeddingVectorizer(
            step=3, model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dmc"
        ),
        # dmm
        typed_ngram_overlap_wt_dmm=author2vec.TfidfEmbeddingVectorizer(
            step=1, model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dmm"
        ),
        typed_ngram_partial_wt_dmm=author2vec.TfidfEmbeddingVectorizer(
            step=2, model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dmm"
        ),
        typed_ngram_non_overlap_wt_dmm=author2vec.TfidfEmbeddingVectorizer(
            step=3, model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dmm"
        ),
        # dbow
        typed_ngram_overlap_wt_dbow=author2vec.TfidfEmbeddingVectorizer(
            step=1, model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dbow"
        ),
        typed_ngram_partial_wt_dbow=author2vec.TfidfEmbeddingVectorizer(
            step=2, model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dbow"
        ),
        typed_ngram_non_overlap_wt_dbow=author2vec.TfidfEmbeddingVectorizer(
            step=3, model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dbow"
        ),
        ##########################################################################
        # Normalized book represented as idf*ngram emb of typed ngram embedding
        # dmc
        norm_typed_ngram_overlap_wt_dmc=author2vec.TfidfEmbeddingVectorizer(
            step=1, model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dmc", norm="l2"
        ),
        norm_typed_ngram_partial_wt_dmc=author2vec.TfidfEmbeddingVectorizer(
            step=2, model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dmc", norm="l2"
        ),
        norm_typed_ngram_non_overlap_wt_dmc=author2vec.TfidfEmbeddingVectorizer(
            step=3, model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dmc", norm="l2"
        ),
        norm_typed_ngram_overlap_no_annotate_wt_dmc=author2vec.TfidfEmbeddingVectorizer(
            step=1,
            model_dir=app.AUTHOR2VEC_OVERLAP_NO_ANNOTATE,
            model_name="dmc",
            norm="l2",
        ),
        # dmm
        norm_typed_ngram_overlap_wt_dmm=author2vec.TfidfEmbeddingVectorizer(
            step=1, model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dmm", norm="l2"
        ),
        norm_typed_ngram_partial_wt_dmm=author2vec.TfidfEmbeddingVectorizer(
            step=2, model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dmm", norm="l2"
        ),
        norm_typed_ngram_non_overlap_wt_dmm=author2vec.TfidfEmbeddingVectorizer(
            step=3, model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dmm", norm="l2"
        ),
        # dbow
        norm_typed_ngram_overlap_wt_dbow=author2vec.TfidfEmbeddingVectorizer(
            step=1, model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dbow", norm="l2"
        ),
        norm_typed_ngram_partial_wt_dbow=author2vec.TfidfEmbeddingVectorizer(
            step=2, model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dbow", norm="l2"
        ),
        norm_typed_ngram_non_overlap_wt_dbow=author2vec.TfidfEmbeddingVectorizer(
            step=3, model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dbow", norm="l2"
        ),
        ##########################################################################
        # Author Embeddings
        ##########################################################################
        # author emb
        author_overlap_dmc=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dmc"
        ),
        author_overlap_dmm=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dmm"
        ),
        author_overlap_dbow=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dbow"
        ),
        author_partial_dmc=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dmc"
        ),
        author_partial_dmm=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dmm"
        ),
        author_partial_dbow=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dbow"
        ),
        author_non_overlap_dmc=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dmc"
        ),
        author_non_overlap_dmm=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dmm"
        ),
        author_non_overlap_dbow=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dbow"
        ),
        # author emb normalized
        norm_author_overlap_dmc=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dmc", norm="l2"
        ),
        norm_author_overlap_no_annotate_dmc=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_OVERLAP_NO_ANNOTATE, model_name="dmc", norm="l2"
        ),
        norm_author_overlap_dmm=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dmm", norm="l2"
        ),
        norm_author_overlap_dbow=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dbow", norm="l2"
        ),
        norm_author_partial_dmc=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dmc", norm="l2"
        ),
        norm_author_partial_dmm=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dmm", norm="l2"
        ),
        norm_author_partial_dbow=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dbow", norm="l2"
        ),
        norm_author_non_overlap_dmc=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dmc", norm="l2"
        ),
        norm_author_non_overlap_dmm=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dmm", norm="l2"
        ),
        norm_author_non_overlap_dbow=author2vec.Author2VecFeatures(
            model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dbow", norm="l2"
        ),
        ##########################################################################
        # author binary feature
        author_binary=author2vec.AuthorFeatures(),
        # author infer vec for genre classification
        author_infer_overlap_dmc=author2vec.AuthorInferVecFeatures(
            model_dir=app.AUTHOR2VEC_OVERLAP, model_name="dmc", step=1
        ),
        author_infer_partial_dmc=author2vec.AuthorInferVecFeatures(
            model_dir=app.AUTHOR2VEC_PARTIAL, model_name="dmc", step=2
        ),
        author_infer_non_overlap_dmc=author2vec.AuthorInferVecFeatures(
            model_dir=app.AUTHOR2VEC_NON_OVERLAP, model_name="dmc", step=3
        ),
    )

    return features_dic[f_name]


def create_feature(feature_names):
    """Utility function to create features object

    Parameters
    -----------
    feature_names : features name or list of features names


    Returns
    --------
    a tuple of (feature_name, features object)
       lst features names are joined by -
       features object is the union of all features in the lst

    """
    try:
        # print (feature_names)
        if isinstance(feature_names, list):
            return (
                "-".join(feature_names),
                FeatureUnion([(f, get_feature(f)) for f in feature_names]),
            )
        else:

            return (feature_names, get_feature(feature_names))
    except Exception as e:
        print(e)
        raise ValueError("Error in function ")

