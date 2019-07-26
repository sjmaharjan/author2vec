# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from .book import Book
from author2vec.manage import app
import pandas as pd


class AuthorCorpusReader(object):
    def __init__(self, dirname):
        self.dirname = os.path.expanduser(dirname)

    def __iter__(self):
        for author_folder in os.listdir(self.dirname):
            for book_file in os.listdir(os.path.join(self.dirname, author_folder)):
                author_id, author_name = author_folder.split('_', 1)
                book_path = os.path.join(self.dirname, author_folder, book_file)
                book_id, _ = book_file.split('_', 1)
                yield Book(book_path=book_path, book_id=book_id, genre=None, success=None,
                           avg_rating=None, author_id=author_id, author_name=author_name)
