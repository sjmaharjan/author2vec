import codecs
from collections import defaultdict
import click
from .config import config
import os
import sys
import logging
import time

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
# Set the path

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

app = config[os.getenv("BOOKS_CONFIG") or "default"]


@click.group()
def manager():
    pass


@manager.command()
@click.option("--f", help="feature id")
def dump_vectors(f):
    from author2vec.readers.corpus import Corpus
    from author2vec.readers.goodreads import GoodreadsReader
    from author2vec.features.utils import fetch_features_vectorized

    data_dir = os.path.join(basedir, app.DATA)  # for all data change path to data_all
    # data_dir = os.path.join(basedir, '../corpus/data_all') #for all data change path to data_all
    split_file = os.path.join(basedir, app.SPLIT_FILE)
    goodread_reader = GoodreadsReader(dirname=data_dir)
    goodreadscorpus = Corpus.from_splitfile(
        reader=goodread_reader, split_file=split_file
    )
    print("Done loading data")
    # for i, feature in enumerate(app.FEATURES):
    feature = app.FEATURES[int(f)]
    print("Running Feature Extraction for feature  {}".format(feature))
    fetch_features_vectorized(
        data_dir=app.VECTORS, features=feature, corpus=goodreadscorpus
    )

    print("Done....")


@manager.command()
@click.option("--f", help="feature id")
def dump_genre_vectors(f):
    from author2vec.readers.corpus import Corpus
    from author2vec.readers.goodreads import GoodreadsReader
    from author2vec.features.utils import fetch_features_vectorized
    from author2vec.readers.book import genre_le

    data_dir = os.path.join(basedir, app.DATA)
    # split_file = os.path.join(basedir, '../data/train_test_split_goodreads.yaml')
    goodread_reader = GoodreadsReader(dirname=data_dir)
    goodreadscorpus = Corpus.with_splits(
        reader=goodread_reader, label_extractor=genre_le
    )
    print("Done loading data")
    # for i, feature in enumerate(app.FEATURES):
    feature = app.FEATURES[int(f)]
    print("Running Feature Extraction for feature  {}".format(feature))
    fetch_features_vectorized(
        data_dir=app.VECTORS, features=feature, corpus=goodreadscorpus
    )

    print("Done....")


@manager.command()
@click.option("--f", help="feature id")
def run_success(f):
    from author2vec.readers.corpus import Corpus
    from author2vec.readers.goodreads import GoodreadsReader
    from author2vec.experiments.success import success_classification

    data_dir = os.path.join(basedir, app.DATA)
    # data_dir = os.path.join(basedir, '../corpus/data_all')
    split_file = os.path.join(basedir, app.SPLIT_FILE)
    goodread_reader = GoodreadsReader(dirname=data_dir)
    goodreadscorpus = Corpus.from_splitfile(
        reader=goodread_reader, split_file=split_file
    )
    print("Done loading data")
    # for i, feature in enumerate(app.FEATURES):
    feature = app.FEATURES[int(f)]
    print("Running Success Experiment for feature  {}".format(feature))
    success_classification(
        corpus=goodreadscorpus, feature=feature, dump_dir=app.VECTORS, ignore_lst=None
    )

    print("Done....")


@manager.command()
@click.option("--f", help="feature id")
def run_genre(f):
    from author2vec.readers.corpus import Corpus
    from author2vec.readers.goodreads import GoodreadsReader
    from author2vec.readers.book import genre_le
    from author2vec.experiments.genre import genre_classification

    data_dir = os.path.join(basedir, app.DATA)
    # split_file = os.path.join(basedir, '../data/train_test_split_goodreads.yaml')
    goodread_reader = GoodreadsReader(dirname=data_dir)
    goodreadscorpus = Corpus.with_splits(
        reader=goodread_reader, label_extractor=genre_le
    )
    print("Done loading data")
    # for i, feature in enumerate(app.FEATURES):
    feature = app.FEATURES[int(f)]
    print("Running Genre Experiment for feature  {}".format(feature))
    genre_classification(
        corpus=goodreadscorpus, feature=feature, dump_dir=app.VECTORS, ignore_lst=None
    )

    print("Done....")


@manager.command()
@click.option("--f", help="feature id")
def run_success_multitask(f):

    from author2vec.readers.corpus import Corpus
    from author2vec.readers.book import genre_success_le
    from author2vec.readers.goodreads import GoodreadsReader
    from author2vec.experiments.success import success_classification_multitask

    data_dir = os.path.join(basedir, app.DATA)
    # data_dir = os.path.join(basedir, '../corpus/data_all')
    # file='train_test_split_goodreads.yaml'
    # file = 'genre_splits.yml'
    # split_file = os.path.join(basedir, '../data/train_test_split_goodreads.yaml')
    split_file = os.path.join(basedir, app.SPLIT_FILE)
    # split_file = os.path.join(basedir, '../data/train_test_split_goodreads.yaml')
    goodread_reader = GoodreadsReader(dirname=data_dir)
    goodreadscorpus = Corpus.from_splitfile(
        reader=goodread_reader, split_file=split_file, label_extractor=genre_success_le
    )
    print("Done loading data")
    # for i, feature in enumerate(app.FEATURES):
    feature = app.FEATURES[int(f)]
    print("Running Success Experiment for feature  {}".format(feature))
    success_classification_multitask(
        corpus=goodreadscorpus, feature=feature, dump_dir=app.VECTORS, ignore_lst=None
    )

    print("Done....")


@manager.command()
def collect_results():
    from collections import defaultdict
    import pandas as pd

    SUCCESS_OUTPUT = os.path.join(basedir, "../results/normlized/st_300_w5_e100")
    SUCCESS_OUTPUT_MT = os.path.join(basedir, "../results/normlized/mt_300_w5_e100")

    def get_results(fn):
        with open(fn) as f_in:
            # Feature,author_overlap_dbow, weighted F1,0.6989072799282908
            try:
                result = f_in.readlines()
                if not result:
                    print(fn)
                    return None, None, None
                result = result[-1]
            except Exception as e:
                print(fn)

                result = f_in.readlines()[-2]
            f1_score = float(result.split(" weighted F1,")[-1])
            feature_lst = result.split(" weighted F1,")[0].strip(",").split(",")[1:]
            if "non_overlap" in feature_lst[0]:
                type_char_ngarm = "non_overlap"
            elif "overlap" in feature_lst[0]:
                type_char_ngarm = "overlap"
            elif "partial" in feature_lst[0]:
                type_char_ngarm = "partial"
            else:
                # type_char_ngarm = 'overlap'
                raise ValueError("Not defined char ngram typr")
            return (
                "::".join([f.replace(type_char_ngarm, "") for f in feature_lst]),
                type_char_ngarm,
                f1_score,
            )

    st_result = SUCCESS_OUTPUT
    mt_result = SUCCESS_OUTPUT_MT

    result_dic = defaultdict(dict)
    for result_file in os.listdir(st_result):
        st_file = os.path.join(st_result, result_file)
        mt_file = os.path.join(mt_result, result_file.replace(".n.txt", "_n.txt"))

        feature_type, char_ngram_type, f1_score = get_results(st_file)
        if feature_type:
            result_dic[feature_type][char_ngram_type + "_ST"] = f1_score

        feature_type, char_ngram_type, f1_score = get_results(mt_file)
        if feature_type:
            result_dic[feature_type][char_ngram_type + "_MT"] = f1_score
    columns = [
        "overlap_ST",
        "overlap_MT",
        "partial_ST",
        "partial_MT",
        "non_overlap_ST",
        "non_overlap_MT",
    ]

    result_df = pd.DataFrame.from_dict(result_dic, orient="index")
    result_df = result_df[columns]
    result_df.to_csv(
        os.path.join(
            os.path.dirname(st_result),
            os.path.basename(st_result) + "_" + os.path.basename(mt_result) + ".tsv",
        ),
        sep="\t",
    )


if __name__ == "__main__":
    manager()
