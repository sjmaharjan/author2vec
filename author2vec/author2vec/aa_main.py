import os
from collections import Counter
from nltk import sent_tokenize

basedir = os.path.dirname(__file__)


def run_aa(feature_id):
    from author2vec.features import create_feature
    from author2vec.readers.author_corpus_reader import AuthorCorpusReader
    from author2vec.readers.goodreads import GoodreadsReader
    from author2vec.experiments.aa import aa_classification

    n_sentences = 1000

    # features_lst = ['unigram', 'char_tri', ]
    features_lst = [

                    'unigram',
                    'char_tri',

                    'typed_ngram_overlap',
                    'typed_ngram_partial',
                    'typed_ngram_non_overlap',
                    # 'typed_ngram_overlap_no_annotate',

                    # Book Representation

                    'typed_ngram_overlap_mean_dmc',
                    'typed_ngram_partial_mean_dmc',
                    'typed_ngram_non_overlap_mean_dmc',

                    'typed_ngram_overlap_wt_dmc',
                    'typed_ngram_partial_wt_dmc',
                    'typed_ngram_non_overlap_wt_dmc',


                    'typed_ngram_overlap_mean_dmm',
                    'typed_ngram_partial_mean_dmm',
                    'typed_ngram_non_overlap_mean_dmm',

                    'typed_ngram_overlap_wt_dmm',
                    'typed_ngram_partial_wt_dmm',
                    'typed_ngram_non_overlap_wt_dmm',

                    # Book Representaiton normalized
                    'norm_typed_ngram_overlap_mean_dmc',
                    'norm_typed_ngram_partial_mean_dmc',
                    'norm_typed_ngram_non_overlap_mean_dmc',

                    'norm_typed_ngram_overlap_wt_dmc',
                    'norm_typed_ngram_partial_wt_dmc',
                    'norm_typed_ngram_non_overlap_wt_dmc',

                    'norm_typed_ngram_overlap_mean_dmm',
                    'norm_typed_ngram_partial_mean_dmm',
                    'norm_typed_ngram_non_overlap_mean_dmm',

                    'norm_typed_ngram_overlap_wt_dmm',
                    'norm_typed_ngram_partial_wt_dmm',
                    'norm_typed_ngram_non_overlap_wt_dmm',

                    # Author Infer
                    'author_infer_overlap_dmc',
                    'author_infer_partial_dmc',
                    'author_infer_non_overlap_dmc',

                    # Author Infer +Books

                    ['author_infer_overlap_dmc', 'typed_ngram_overlap_mean_dmc'],
                    ['author_infer_partial_dmc', 'typed_ngram_partial_mean_dmc'],
                    ['author_infer_non_overlap_dmc',
                        'typed_ngram_non_overlap_mean_dmc'],


                    ['author_infer_overlap_dmc', 'norm_typed_ngram_overlap_mean_dmc'],
                    ['author_infer_partial_dmc', 'norm_typed_ngram_partial_mean_dmc'],
                    ['author_infer_non_overlap_dmc',
                        'norm_typed_ngram_non_overlap_mean_dmc'],


                    ['author_infer_overlap_dmc', 'norm_typed_ngram_overlap_wt_dmc'],
                    ['author_infer_partial_dmc', 'norm_typed_ngram_partial_wt_dmc'],
                    ['author_infer_non_overlap_dmc',
                        'norm_typed_ngram_non_overlap_wt_dmc'],

                    ['author_infer_overlap_dmc', 'typed_ngram_overlap_wt_dmc'],
                    ['author_infer_partial_dmc', 'typed_ngram_partial_wt_dmc'],
                    ['author_infer_non_overlap_dmc',
                        'typed_ngram_non_overlap_wt_dmc']


                    ]
    f = features_lst[feature_id]

    print("Feature selected {}".format(f))
    #/uhpc/solorio/suraj/authors/author_style/data/AA_auth2vec_5books_v1
    data_dir = os.path.join(basedir, "..", "data", "AA_auth2vec_5books_v1")
    result_dir = os.path.join(basedir, "aa_results")
    author_reader = AuthorCorpusReader(dirname=data_dir)

    books, labels = [], []
    for book in author_reader:
        sub_book = book.of_size(n_sentences)
        books.append(sub_book)
        labels.append(book.author_id)

    print("Done loading data")
    print("Number of instances {}".format(len(books)))

    # print(books[0].content)

    print("Author and books {}".format(Counter(labels)))

    feature_name, feature_extractor = create_feature(f)

    print("Running Success Experiment for feature  {}".format(feature_name))

    out_file = os.path.join(result_dir, '{}.r'.format(feature_name))
    aa_classification(books=books, labels=labels,
                      feature_extractor=feature_extractor, features=feature_name, out_file=out_file)

    print('Done....')


def get_common_books(aa_book_path, external_corpus_path):
    from collections import defaultdict
    import pandas as pd
    aa_books = defaultdict(list)
    external_books = defaultdict(list)
    # aa
    for author_folder in os.listdir(aa_book_path):
            for book_file in os.listdir(os.path.join(aa_book_path, author_folder)):
                author_id, author_name = author_folder.split('_', 1)
                book_id, _ = book_file.split('_', 1)
                aa_books[author_id].append(book_id)

    # external
    for author_folder in os.listdir(external_corpus_path):
            for book_file in os.listdir(os.path.join(external_corpus_path, author_folder, 'cleaned')):

                book_id, _ = book_file.split('.', 1)
                external_books[author_folder].append(book_id)

    final_results = []
    for author in aa_books.keys():
        if author in external_books:
            final_results.append({'author_id': author,
                                 'aa_books': aa_books[author],
                                'ext_books': external_books[author],
                                'common': set(aa_books[author]) & set(external_books[author]),
                                'common_len': len(set(aa_books[author]) & set(external_books[author]))})

    df=pd.DataFrame(final_results)
    df.to_pickle('common_books.pkl')


if __name__ == "__main__":
    
    # aa_book_path='/uhpc/solorio/suraj/authors/author_style/data/AA_auth2vec_5books/'

    # external_corpus_path='/uhpc/solorio/suraj/authors/author_style/data/authors'

    # get_common_books(aa_book_path,external_corpus_path)
    
    import sys
    feature_id=int(sys.argv[1])

    run_aa(feature_id)
