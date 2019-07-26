from gensim.models.doc2vec import TaggedDocument, Doc2Vec, FAST_VERSION
from os import path
import os
from collections import OrderedDict
import datetime
import joblib
from random import shuffle


class GutenbergAuthor(object):
    """
     data arranged in folders
     each folder to have at most 5 book by the same author
     folder id are author ids

    """

    def __init__(self, loc, char_type):
        self.loc = loc
        self.char_type = char_type

    def read(self, fn):
        return joblib.load(fn)

    def __iter__(self):
        for author in os.listdir(self.loc):
            if path.isdir(path.join(self.loc, author)):
                for book in os.listdir(path.join(self.loc, author, self.char_type)):
                    book_name = path.join(self.loc, author, self.char_type, book)
                    content_lst = self.read(book_name)
                    if content_lst:
                        yield TaggedDocument(content_lst, ["Author_{}".format(author)])


def author_style_emb_train(
    model_name,
    base_dir,
    model_dir,
    char_type,
    window=5,
    author_emb_size=300,
    epochs=100,
):
    cores = 10

    corpus = list(doc for doc in GutenbergAuthor(loc=base_dir, char_type=char_type))

    if model_name == "dmc":
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        model = Doc2Vec(
            dm=1,
            dm_concat=1,
            size=author_emb_size,
            window=window,
            negative=5,
            hs=0,
            min_count=2,
            sample=1e-5,
            workers=cores,
        )
    elif model_name == "dbow":
        # PV-DBOW
        model = Doc2Vec(
            dm=0,
            window=window,
            size=author_emb_size,
            negative=5,
            hs=0,
            min_count=2,
            sample=1e-5,
            workers=cores,
        )
    elif model_name == "dmm":
        # PV-DM w/average
        model = Doc2Vec(
            dm=1,
            dm_mean=1,
            size=author_emb_size,
            window=window,
            negative=5,
            hs=0,
            min_count=2,
            sample=1e-5,
            workers=cores,
        )

    else:
        raise ValueError("No model found!!!")

    # # speed setup by sharing results of 1st model's vocabulary scan
    model.build_vocab(
        corpus
    )  # PV-DM/concat requires one special NULL word so it serves as template
    # print(simple_models[0])
    # for model in simple_models[1:]:
    #     model.reset_from(simple_models[0])
    #     print(model)

    # models_by_name = OrderedDict((name, model) for name, model in zip(['dmc', 'dbow', 'dmm'], simple_models))

    print("START %s" % datetime.datetime.now())

    alpha, min_alpha = (0.025, 0.001)
    alpha_delta = (alpha - min_alpha) / epochs
    print("Model: {}".format(model_name))
    for epoch in range(epochs):
        shuffle(corpus)  # shuffling gets best results
        # train
        print("Epoch: {}".format(epoch))
        model.alpha, model.min_alpha = alpha, alpha
        model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
        print("completed pass %i at alpha %f" % (epoch + 1, alpha))
        alpha -= alpha_delta
    print("Saving model {}".format(model_name))
    model.save(model_dir + "/" + "model_%s.doc2vec" % model_name)
    model.save_word2vec_format(
        model_dir + "/" + "word2vec_model_%s.doc2vec" % model_name
    )

    print("END %s" % str(datetime.datetime.now()))


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    jobid = int(argv[0])
    location = "/uhpc/solorio/suraj/authors/author_style/data1/"
    model_dir = (
        "/uhpc/solorio/suraj/authors/author_style/author_embeddings_300_w5_e100/"
    )
    if jobid in [0, 1, 2]:
        model_dir = os.path.join(model_dir, "overlap")

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if jobid == 0:
            author_style_emb_train(
                model_name="dmc",
                base_dir=location,
                model_dir=model_dir,
                char_type="overlap",
            )
        elif jobid == 1:

            author_style_emb_train(
                model_name="dbow",
                base_dir=location,
                model_dir=model_dir,
                char_type="overlap",
            )
        elif jobid == 2:
            author_style_emb_train(
                model_name="dmm",
                base_dir=location,
                model_dir=model_dir,
                char_type="overlap",
            )

    elif jobid in [3, 4, 5]:
        model_dir = os.path.join(model_dir, "partial")
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if jobid == 3:
            author_style_emb_train(
                model_name="dmc",
                base_dir=location,
                model_dir=model_dir,
                char_type="partial",
            )
        elif jobid == 4:

            author_style_emb_train(
                model_name="dbow",
                base_dir=location,
                model_dir=model_dir,
                char_type="partial",
            )
        elif jobid == 5:
            author_style_emb_train(
                model_name="dmm",
                base_dir=location,
                model_dir=model_dir,
                char_type="partial",
            )

    elif jobid in [6, 7, 8]:
        model_dir = os.path.join(model_dir, "non_overlap")

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if jobid == 6:
            author_style_emb_train(
                model_name="dmc",
                base_dir=location,
                model_dir=model_dir,
                char_type="non_overlap",
            )
        elif jobid == 7:

            author_style_emb_train(
                model_name="dbow",
                base_dir=location,
                model_dir=model_dir,
                char_type="non_overlap",
            )
        elif jobid == 8:
            author_style_emb_train(
                model_name="dmm",
                base_dir=location,
                model_dir=model_dir,
                char_type="non_overlap",
            )

    elif jobid in [9, 10, 11]:
        model_dir = os.path.join(model_dir, "overlap_no_annotation")

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if jobid == 9:
            author_style_emb_train(
                model_name="dmc",
                base_dir=location,
                model_dir=model_dir,
                char_type="overlap_no_annotation",
            )
        elif jobid == 10:

            author_style_emb_train(
                model_name="dbow",
                base_dir=location,
                model_dir=model_dir,
                char_type="overlap_no_annotation",
            )
        elif jobid == 11:
            author_style_emb_train(
                model_name="dmm",
                base_dir=location,
                model_dir=model_dir,
                char_type="overlap_no_annotation",
            )

    else:
        print("Invalid jobid")
