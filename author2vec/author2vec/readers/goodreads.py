import os
from .book import Book
from author2vec.manage import app
import pandas as pd


class GoodreadsReader(object):
    def __init__(
        self,
        dirname,
        genres=None,
        meta_file=app.BOOK_META_INFO,
        meta_author=app.AUTHOR_META_INFO,
    ):
        self.dirname = dirname
        if genres is not None:
            self.genres = genres
        else:
            self.genres = sorted(
                [
                    "Fiction",
                    "Science_fiction",
                    "Detective_and_mystery_stories",
                    "Poetry",
                    "Short_stories",
                    "Love_stories",
                    "Historical_fiction",
                    "Drama",
                ]
            )
        self.book_df = pd.read_excel(meta_file)
        self.author_df = pd.read_csv(meta_author, sep="\t")

    def __iter__(self):
        for genre in self.genres:
            for category in ["failure", "success"]:
                for fid in os.listdir(os.path.join(self.dirname, genre, category)):
                    fname = os.path.join(self.dirname, genre, category, fid)
                    if fid.startswith(".DS_Store") or not os.path.isfile(fname):
                        continue
                    if category.startswith("failure"):
                        success = 0
                    else:
                        success = 1
                    avg_rating = self.book_df[self.book_df["FILENAME"] == fid][
                        "AVG_RATING_2016"
                    ].values[0]
                    author_id = self.author_df[self.author_df["FILENAME"] == fid][
                        "Author_id"
                    ].values[0]
                    author_name = self.author_df[self.author_df["FILENAME"] == fid][
                        "Author"
                    ].values[0]

                    yield Book(
                        book_path=fname,
                        book_id=fid,
                        genre=genre,
                        success=success,
                        avg_rating=round(avg_rating, 3),
                        author_id=author_id,
                        author_name=author_name,
                    )

