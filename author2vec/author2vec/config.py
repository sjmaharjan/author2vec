import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True

    DATA='/uhpc/solorio/suraj/Books/data'
    SPLIT_FILE='../data/train_test_split_goodreads.yaml'

    FEATURES = [
        # 'author_infer_overlap_dmc',
        # 'author_infer_partial_dmc',
        # 'author_infer_non_overlap_dmc',
        # 'norm_author_overlap_no_annotate_dmc',
        #'norm_author_overlap_no_annotate_dmc',
        #'norm_typed_ngram_overlap_no_annotate_wt_dmc',


        ['author_binary','typed_ngram_overlap'],
        ['author_binary', 'typed_ngram_partial'],
        ['author_binary', 'typed_ngram_non_overlap'],


       # ['norm_author_overlap_no_annotate_dmc',  'norm_typed_ngram_overlap_no_annotate_wt_dmc'],

        # 'author_binary',
       #  'author_overlap_dmc',
       #  'author_overlap_dmm',
       #  'author_overlap_dbow',
       #  'author_partial_dmc',
       #  'author_partial_dmm',
       #  'author_partial_dbow',
       #  'author_non_overlap_dmc',
       #  'author_non_overlap_dmm',
       #  'author_non_overlap_dbow',
       #
       #  #Type ngram
       #  'typed_ngram_overlap',
       #  'typed_ngram_partial',
       #  'typed_ngram_non_overlap',
       #
       #  #Mean
       #  # dmc
       #  'typed_ngram_overlap_mean_dmc',
       #  'typed_ngram_partial_mean_dmc',
       #  'typed_ngram_non_overlap_mean_dmc',
       #  # dmm
       #  'typed_ngram_overlap_mean_dmm',
       #  'typed_ngram_partial_mean_dmm',
       #  'typed_ngram_non_overlap_mean_dmm',
       #  # dbow
       #  'typed_ngram_overlap_mean_dbow',
       #  'typed_ngram_partial_mean_dbow',
       #  'typed_ngram_non_overlap_mean_dbow',
       #
       #
       #  #TFIDF
       #  # dmc
       #  'typed_ngram_overlap_wt_dmc',
       #  'typed_ngram_partial_wt_dmc',
       #  'typed_ngram_non_overlap_wt_dmc',
       #  # ',dmm
       #  'typed_ngram_overlap_wt_dmm',
       #  'typed_ngram_partial_wt_dmm',
       #  'typed_ngram_non_overlap_wt_dmm',
       #  # dbow
       #  'typed_ngram_overlap_wt_dbow',
       #  'typed_ngram_partial_wt_dbow',
       #  'typed_ngram_non_overlap_wt_dbow',
       #
       #
       #  #Normallized Features
       #
       #  'norm_author_overlap_dmc',
       #  'norm_author_overlap_dmm',
       #  'norm_author_overlap_dbow',
       #  'norm_author_partial_dmc',
       #  'norm_author_partial_dmm',
       #  'norm_author_partial_dbow',
       #  'norm_author_non_overlap_dmc',
       #  'norm_author_non_overlap_dmm',
       #  'norm_author_non_overlap_dbow',
       #
       #  # Mean
       #  # dmc
       #  'norm_typed_ngram_overlap_mean_dmc',
       #  'norm_typed_ngram_partial_mean_dmc',
       #  'norm_typed_ngram_non_overlap_mean_dmc',
       #  # dmm
       #  'norm_typed_ngram_overlap_mean_dmm',
       #  'norm_typed_ngram_partial_mean_dmm',
       #  'norm_typed_ngram_non_overlap_mean_dmm',
       #  # dbow
       #  'norm_typed_ngram_overlap_mean_dbow',
       #  'norm_typed_ngram_partial_mean_dbow',
       #  'norm_typed_ngram_non_overlap_mean_dbow',
       #
       #  # TFIDF
       #  # dmc
       #  'norm_typed_ngram_overlap_wt_dmc',
       #  'norm_typed_ngram_partial_wt_dmc',
       #  'norm_typed_ngram_non_overlap_wt_dmc',
       #  # ',dmm
       #  'norm_typed_ngram_overlap_wt_dmm',
       #  'norm_typed_ngram_partial_wt_dmm',
       #  'norm_typed_ngram_non_overlap_wt_dmm',
       #  # dbow
       #  'norm_typed_ngram_overlap_wt_dbow',
       #  'norm_typed_ngram_partial_wt_dbow',
       #  'norm_typed_ngram_non_overlap_wt_dbow',
       #
       #
       #
       #
       #
       #  ###### combine corresponding author emb ##########
       #
       #
       # # Mean
       #  # dmc
       #  ['typed_ngram_overlap_mean_dmc','author_overlap_dmc'],
       #  ['typed_ngram_partial_mean_dmc','author_partial_dmc'],
       #  ['typed_ngram_non_overlap_mean_dmc', 'author_non_overlap_dmc'],
       #  # dmm
       #  ['typed_ngram_overlap_mean_dmm','author_overlap_dmm'],
       #  ['typed_ngram_partial_mean_dmm','author_partial_dmm'],
       #  ['typed_ngram_non_overlap_mean_dmm','author_non_overlap_dmm'],
       #  # dbow
       #  ['typed_ngram_overlap_mean_dbow','author_overlap_dbow'],
       #  ['typed_ngram_partial_mean_dbow','author_partial_dbow'],
       #  ['typed_ngram_non_overlap_mean_dbow','author_non_overlap_dbow'],
       #
       #  # TF-IDF weights
       #  # dmc
       #  ['typed_ngram_overlap_wt_dmc', 'author_overlap_dmc'],
       #  ['typed_ngram_partial_wt_dmc', 'author_partial_dmc'],
       #  ['typed_ngram_non_overlap_wt_dmc', 'author_non_overlap_dmc'],
       #  # dmm
       #  ['typed_ngram_overlap_wt_dmm', 'author_overlap_dmm'],
       #  ['typed_ngram_partial_wt_dmm', 'author_partial_dmm'],
       #  ['typed_ngram_non_overlap_wt_dmm', 'author_non_overlap_dmm'],
       #  # dbow
       #  ['typed_ngram_overlap_wt_dbow', 'author_overlap_dbow'],
       #  ['typed_ngram_partial_wt_dbow', 'author_partial_dbow'],
       #  ['typed_ngram_non_overlap_wt_dbow', 'author_non_overlap_dbow'],
       #
       #
       #  # with author and type ngram
       #
       #  ['typed_ngram_overlap', 'author_overlap_dmc'],
       #  ['typed_ngram_overlap', 'author_overlap_dmm'],
       #  ['typed_ngram_overlap', 'author_overlap_dbow'],
       #  ['typed_ngram_partial', 'author_partial_dmc'],
       #  ['typed_ngram_partial', 'author_partial_dmm'],
       #  ['typed_ngram_partial', 'author_partial_dbow'],
       #  ['typed_ngram_non_overlap', 'author_non_overlap_dmc'],
       #  ['typed_ngram_non_overlap', 'author_non_overlap_dmm'],
       #  ['typed_ngram_non_overlap', 'author_non_overlap_dbow'],
       #
       # #
       # #
       # #
       # #  # combination with normalized features
       # #
       # #
       #  # Mean
       #  # dmc
       #  ['norm_typed_ngram_overlap_mean_dmc', 'norm_author_overlap_dmc'],
       #  ['norm_typed_ngram_partial_mean_dmc', 'norm_author_partial_dmc'],
       #  ['norm_typed_ngram_non_overlap_mean_dmc', 'norm_author_non_overlap_dmc'],
       #  # dmm
       #  ['norm_typed_ngram_overlap_mean_dmm', 'norm_author_overlap_dmm'],
       #  ['norm_typed_ngram_partial_mean_dmm', 'norm_author_partial_dmm'],
       #  ['norm_typed_ngram_non_overlap_mean_dmm', 'norm_author_non_overlap_dmm'],
       #  # dbow
       #  ['norm_typed_ngram_overlap_mean_dbow', 'norm_author_overlap_dbow'],
       #  ['norm_typed_ngram_partial_mean_dbow', 'norm_author_partial_dbow'],
       #  ['norm_typed_ngram_non_overlap_mean_dbow', 'norm_author_non_overlap_dbow'],
       #
       #  # TF-IDF weights
       #  # dmc
       #  ['norm_typed_ngram_overlap_wt_dmc', 'norm_author_overlap_dmc'],
       #  ['norm_typed_ngram_partial_wt_dmc', 'norm_author_partial_dmc'],
       #  ['norm_typed_ngram_non_overlap_wt_dmc', 'norm_author_non_overlap_dmc'],
       #  # dmm
       #  ['norm_typed_ngram_overlap_wt_dmm', 'norm_author_overlap_dmm'],
       #  ['norm_typed_ngram_partial_wt_dmm', 'norm_author_partial_dmm'],
       #  ['norm_typed_ngram_non_overlap_wt_dmm', 'norm_author_non_overlap_dmm'],
       #  # dbow
       #  ['norm_typed_ngram_overlap_wt_dbow', 'norm_author_overlap_dbow'],
       #  ['norm_typed_ngram_partial_wt_dbow', 'norm_author_partial_dbow'],
       #  ['norm_typed_ngram_non_overlap_wt_dbow', 'norm_author_non_overlap_dbow'],
       #
       #  # with author and type ngram
       #
       #  ['typed_ngram_overlap', 'norm_author_overlap_dmc'],
       #  ['typed_ngram_overlap', 'norm_author_overlap_dmm'],
       #  ['typed_ngram_overlap', 'norm_author_overlap_dbow'],
       #  ['typed_ngram_partial', 'norm_author_partial_dmc'],
       #  ['typed_ngram_partial', 'norm_author_partial_dmm'],
       #  ['typed_ngram_partial', 'norm_author_partial_dbow'],
       #  ['typed_ngram_non_overlap', 'norm_author_non_overlap_dmc'],
       #  ['typed_ngram_non_overlap', 'norm_author_non_overlap_dmm'],
       #  ['typed_ngram_non_overlap', 'norm_author_non_overlap_dbow'],

        #Author binary features

         # dmc
         # ['norm_typed_ngram_overlap_mean_dmc', 'author_binary'],
         # ['norm_typed_ngram_partial_mean_dmc', 'author_binary'],
         # ['norm_typed_ngram_non_overlap_mean_dmc', 'author_binary'],
         # # dmm
         # ['norm_typed_ngram_overlap_mean_dmm', 'author_binary'],
         # ['norm_typed_ngram_partial_mean_dmm', 'author_binary'],
         # ['norm_typed_ngram_non_overlap_mean_dmm', 'author_binary'],
         # # dbow
         # ['norm_typed_ngram_overlap_mean_dbow', 'author_binary'],
         # ['norm_typed_ngram_partial_mean_dbow', 'author_binary'],
         # ['norm_typed_ngram_non_overlap_mean_dbow', 'author_binary'],
         #
         # # TF-IDF weights
         # # dmc
         # ['norm_typed_ngram_overlap_wt_dmc', 'author_binary'],
         # ['norm_typed_ngram_partial_wt_dmc', 'author_binary'],
         # ['norm_typed_ngram_non_overlap_wt_dmc', 'author_binary'],
         # # dmm
         # ['norm_typed_ngram_overlap_wt_dmm', 'author_binary'],
         # ['norm_typed_ngram_partial_wt_dmm', 'author_binary'],
         # ['norm_typed_ngram_non_overlap_wt_dmm', 'author_binary'],
         # # dbow
         # ['norm_typed_ngram_overlap_wt_dbow', 'author_binary'],
         # ['norm_typed_ngram_partial_wt_dbow', 'author_binary'],
         # ['norm_typed_ngram_non_overlap_wt_dbow', 'author_binary'],
         #
         # # with author and type ngram
         #
         # ['typed_ngram_overlap', 'author_binary'],
         #
         # ['typed_ngram_partial', 'author_binary'],
         #
         # ['typed_ngram_non_overlap', 'author_binary'],



        #
        # ###NEW EXPERIMENTS
        # #dmc+dbow
        #
        #
        # ['typed_ngram_overlap', 'norm_author_overlap_dmc','norm_author_overlap_dbow'],
        # ['typed_ngram_overlap', 'norm_author_overlap_dmm','norm_author_overlap_dbow'],
        #
        # ['typed_ngram_partial', 'norm_author_partial_dmc','norm_author_partial_dbow'],
        # ['typed_ngram_partial', 'norm_author_partial_dmm','norm_author_partial_dbow'],
        #
        # ['typed_ngram_non_overlap', 'norm_author_non_overlap_dmc','norm_author_non_overlap_dbow'],
        # ['typed_ngram_non_overlap', 'norm_author_non_overlap_dmm','norm_author_non_overlap_dbow'],
        #
        #
        # #mean
        #
        # ['norm_typed_ngram_overlap_mean_dmc', 'norm_typed_ngram_overlap_mean_dbow'],
        # ['norm_typed_ngram_partial_mean_dmc', 'norm_typed_ngram_partial_mean_dbow'],
        # ['norm_typed_ngram_non_overlap_mean_dmc', 'norm_typed_ngram_non_overlap_mean_dbow'],
        #
        # ['norm_typed_ngram_overlap_mean_dmm', 'norm_typed_ngram_overlap_mean_dbow'],
        # ['norm_typed_ngram_partial_mean_dmm', 'norm_typed_ngram_partial_mean_dbow'],
        # ['norm_typed_ngram_non_overlap_mean_dmm', 'norm_typed_ngram_non_overlap_mean_dbow'],
        #
        #
        # #author
        #
        # ['norm_typed_ngram_overlap_mean_dmc', 'norm_typed_ngram_overlap_mean_dbow','norm_author_overlap_dmc','norm_author_overlap_dbow'],
        # ['norm_typed_ngram_partial_mean_dmc', 'norm_typed_ngram_partial_mean_dbow','norm_author_partial_dmc','norm_author_partial_dbow'],
        # ['norm_typed_ngram_non_overlap_mean_dmc', 'norm_typed_ngram_non_overlap_mean_dbow','norm_author_non_overlap_dmc','norm_author_non_overlap_dbow'],
        #
        # ['norm_typed_ngram_overlap_mean_dmm', 'norm_typed_ngram_overlap_mean_dbow','norm_author_overlap_dmm','norm_author_overlap_dbow'],
        # ['norm_typed_ngram_partial_mean_dmm', 'norm_typed_ngram_partial_mean_dbow','norm_author_partial_dmm','norm_author_partial_dbow'],
        # ['norm_typed_ngram_non_overlap_mean_dmm', 'norm_typed_ngram_non_overlap_mean_dbow','norm_author_non_overlap_dmm','norm_author_non_overlap_dbow'],
        #
        #
        # #tfidf
        # ['norm_typed_ngram_overlap_wt_dmc','norm_typed_ngram_overlap_wt_dbow'],
        # ['norm_typed_ngram_partial_wt_dmc','norm_typed_ngram_partial_wt_dbow'],
        # ['norm_typed_ngram_non_overlap_wt_dmc','norm_typed_ngram_non_overlap_wt_dbow'],
        #
        # ['norm_typed_ngram_overlap_wt_dmm', 'norm_typed_ngram_overlap_wt_dbow'],
        # ['norm_typed_ngram_partial_wt_dmm', 'norm_typed_ngram_partial_wt_dbow'],
        # ['norm_typed_ngram_non_overlap_wt_dmm', 'norm_typed_ngram_non_overlap_wt_dbow'],
        #
        # #author
        # ['norm_typed_ngram_overlap_wt_dmc', 'norm_typed_ngram_overlap_wt_dbow','norm_author_overlap_dmc','norm_author_overlap_dbow'],
        # ['norm_typed_ngram_partial_wt_dmc', 'norm_typed_ngram_partial_wt_dbow','norm_author_partial_dmc','norm_author_partial_dbow'],
        # ['norm_typed_ngram_non_overlap_wt_dmc', 'norm_typed_ngram_non_overlap_wt_dbow','norm_author_non_overlap_dmc','norm_author_non_overlap_dbow'],
        #
        # ['norm_typed_ngram_overlap_wt_dmm', 'norm_typed_ngram_overlap_wt_dbow','norm_author_overlap_dmm','norm_author_overlap_dbow'],
        # ['norm_typed_ngram_partial_wt_dmm', 'norm_typed_ngram_partial_wt_dbow','norm_author_partial_dmm','norm_author_partial_dbow'],
        # ['norm_typed_ngram_non_overlap_wt_dmm', 'norm_typed_ngram_non_overlap_wt_dbow','norm_author_non_overlap_dmm','norm_author_non_overlap_dbow'],

    ]


    BOOK_META_INFO = os.path.join(basedir, '../data/gutenberg_goodread_2016_match.xlsx')
    AUTHOR_META_INFO=os.path.join(basedir, '../data/eacl_meta_books_authors_wo_dup.tsv')

    # for 100 dimention vectors

    # VECTORS = os.path.join(basedir, '../vectors')
    #
    # AUTHOR2VEC_OVERLAP = os.path.join(basedir, '../author_embeddings/overlap')
    # AUTHOR2VEC_PARTIAL = os.path.join(basedir, '../author_embeddings/partial')
    # AUTHOR2VEC_NON_OVERLAP = os.path.join(basedir, '../author_embeddings/non_overlap')
    #
    # SUCCESS_OUTPUT = os.path.join(basedir, '../results/normlized/st')
    # SUCCESS_OUTPUT_MT = os.path.join(basedir, '../results/normlized/mt')


    # for 300 dimention vectors

    # VECTORS = os.path.join(basedir, '../vectors_300')
    #
    # AUTHOR2VEC_OVERLAP=os.path.join(basedir, '../author_embeddings_300/overlap')
    # AUTHOR2VEC_PARTIAL=os.path.join(basedir, '../author_embeddings_300/partial')
    # AUTHOR2VEC_NON_OVERLAP=os.path.join(basedir, '../author_embeddings_300/non_overlap')
    #
    # SUCCESS_OUTPUT = os.path.join(basedir, '../results/normlized/st_300')
    # SUCCESS_OUTPUT_MT = os.path.join(basedir, '../results/normlized/mt_300')

    #for 500 dimention vectors

    # VECTORS = os.path.join(basedir, '../vectors_500')
    #
    # AUTHOR2VEC_OVERLAP = os.path.join(basedir, '../author_embeddings_500/overlap')
    # AUTHOR2VEC_PARTIAL = os.path.join(basedir, '../author_embeddings_500/partial')
    # AUTHOR2VEC_NON_OVERLAP = os.path.join(basedir, '../author_embeddings_500/non_overlap')
    #
    # SUCCESS_OUTPUT = os.path.join(basedir, '../results/normlized/st_500')
    # SUCCESS_OUTPUT_MT = os.path.join(basedir, '../results/normlized/mt_500')



    #for author emb with window 10 and epoch 100
    #100

    # VECTORS = os.path.join(basedir, '../vectors_100_w10_e100')
    #
    # AUTHOR2VEC_OVERLAP=os.path.join(basedir, '../author_embeddings_100_w10_e100/overlap')
    # AUTHOR2VEC_PARTIAL=os.path.join(basedir, '../author_embeddings_100_w10_e100/partial')
    # AUTHOR2VEC_NON_OVERLAP=os.path.join(basedir, '../author_embeddings_100_w10_e100/non_overlap')
    #
    # SUCCESS_OUTPUT = os.path.join(basedir, '../results/normlized/st_100_w10_e100')
    # SUCCESS_OUTPUT_MT = os.path.join(basedir, '../results/normlized/mt_100_w10_e100')
    #
    #
    # #300
    #
    # VECTORS = os.path.join(basedir, '../vectors_300_w10_e100')
    #
    # AUTHOR2VEC_OVERLAP = os.path.join(basedir, '../author_embeddings_300_w10_e100/overlap')
    # AUTHOR2VEC_PARTIAL = os.path.join(basedir, '../author_embeddings_300_w10_e100/partial')
    # AUTHOR2VEC_NON_OVERLAP = os.path.join(basedir, '../author_embeddings_300_w10_e100/non_overlap')
    #
    # SUCCESS_OUTPUT = os.path.join(basedir, '../results/normlized/st_300_w10_e100')
    # SUCCESS_OUTPUT_MT = os.path.join(basedir, '../results/normlized/mt_300_w10_e100')
    #
    # #500 next st mt for this
    #
    # VECTORS = os.path.join(basedir, '../vectors_500_w10_e100')
    #
    # AUTHOR2VEC_OVERLAP = os.path.join(basedir, '../author_embeddings_500_w10_e100/overlap')
    # AUTHOR2VEC_PARTIAL = os.path.join(basedir, '../author_embeddings_500_w10_e100/partial')
    # AUTHOR2VEC_NON_OVERLAP = os.path.join(basedir, '../author_embeddings_500_w10_e100/non_overlap')
    #
    # SUCCESS_OUTPUT = os.path.join(basedir, '../results/normlized/st_500_w10_e100')
    # SUCCESS_OUTPUT_MT = os.path.join(basedir, '../results/normlized/mt_500_w10_e100')





    # For windo 5 with 100 epoch

    # #100
    # VECTORS = os.path.join(basedir, '../vectors_100_w5_e100')
    #
    # AUTHOR2VEC_OVERLAP = os.path.join(basedir, '../author_embeddings_100_w5_e100/overlap')
    # AUTHOR2VEC_PARTIAL = os.path.join(basedir, '../author_embeddings_100_w5_e100/partial')
    # AUTHOR2VEC_NON_OVERLAP = os.path.join(basedir, '../author_embeddings_100_w5_e100/non_overlap')
    #
    # SUCCESS_OUTPUT = os.path.join(basedir, '../results/normlized/st_100_w5_e100')
    # SUCCESS_OUTPUT_MT = os.path.join(basedir, '../results/normlized/mt_100_w5_e100')


    # 300
    VECTORS = os.path.join(basedir, '../vectors_300_w5_e100')
    # VECTORS = os.path.join(basedir, '../vectors_genre_300_w5_e100')

    AUTHOR2VEC_OVERLAP = os.path.join(basedir, '../author_embeddings_300_w5_e100/overlap')
    AUTHOR2VEC_PARTIAL = os.path.join(basedir, '../author_embeddings_300_w5_e100/partial')
    AUTHOR2VEC_NON_OVERLAP = os.path.join(basedir, '../author_embeddings_300_w5_e100/non_overlap')
    GENRE_OUTPUT=os.path.join(basedir, '../results/genre/st_300_w5_e100')
    SUCCESS_OUTPUT = os.path.join(basedir, '../results/author_1hot_annotations/')
    SUCCESS_OUTPUT_MT = os.path.join(basedir, '../results/author_1hot_annotations/')

    AUTHOR2VEC_OVERLAP_NO_ANNOTATE = os.path.join(basedir, '../author_embeddings_300_w5_e100/overlap_no_annotation')


    # 500
    # VECTORS = os.path.join(basedir, '../vectors_500_w5_e100')
    #
    # AUTHOR2VEC_OVERLAP = os.path.join(basedir, '../author_embeddings_500_w5_e100/overlap')
    # AUTHOR2VEC_PARTIAL = os.path.join(basedir, '../author_embeddings_500_w5_e100/partial')
    # AUTHOR2VEC_NON_OVERLAP = os.path.join(basedir, '../author_embeddings_500_w5_e100/non_overlap')
    #
    # SUCCESS_OUTPUT = os.path.join(basedir, '../results/normlized/st_500_w5_e100')
    # SUCCESS_OUTPUT_MT = os.path.join(basedir, '../results/normlized/mt_500_w5_e100')



class TestingConfig(Config):
    TESTING = True


class ProductionConfig(Config):
    pass


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
