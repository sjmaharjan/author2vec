import re
import string
import os
import joblib
import tqdm

### REF: https://github.com/explosion/sense2vec/blob/master/bin/merge_text.py

LABELS = {"PREFIX": "0", "SUFFIX": "1", "WHOLE_WORD": "2"}


def represent_ngram(text, tag_type):
    tag = LABELS.get(tag_type, "")
    return text + "|" + tag


##REGEX function to match type

_slash_W = string.punctuation + " " + "”" + "“"

_white_spaces = re.compile(r"\s\s+")

## inline function to match whole_word, prefix, suffix

_whole_word = (
    lambda x, y, i, n: not (re.findall(r"(?:\W|\s)", x))
    and (i == 0 or y[i - 1] in _slash_W)
    and (i + n == len(y) or y[i + n] in _slash_W)
)

_prefix = (
    lambda x, y, i, n: not (re.findall(r"(?:\W|\s)", x))
    and (i == 0 or y[i - 1] in _slash_W)
    and (not (i + n == len(y) or y[i + n] in _slash_W))
)

_suffix = (
    lambda x, y, i, n: not (re.findall(r"(?:\W|\s)", x))
    and (not (i == 0 or y[i - 1] in _slash_W))
    and (i + n == len(y) or y[i + n] in _slash_W)
)


def read_book(fn):
    with open(fn, mode="r", encoding="latin1") as infile:
        content = " ".join([line.replace("\r\n", "") for line in infile.readlines()])
        return content


def transform_doc(doc, n=3, step=1, annotate=True):
    text_document = _white_spaces.sub(
        " ", doc
    )  ## replace multiple spaces with a single space

    text_len = len(text_document)

    ngrams = []

    for i in range(0, text_len - n + 1, step):
        gram = text_document[i : i + n]
        # print(gram)

        if annotate:
            if _whole_word(gram, text_document, i, n):
                ngrams.append(represent_ngram(gram, "WHOLE_WORD"))

            elif _prefix(gram, text_document, i, n):
                ngrams.append(represent_ngram(gram, "PREFIX"))

            elif _suffix(gram, text_document, i, n):
                ngrams.append(represent_ngram(gram, "SUFFIX"))

            else:
                ngrams.append(gram)
        else:
            ngrams.append(gram)

    return ngrams


def load_transform_dump(loc, step, annotate=True):
    def get_folder_name():
        step_folder_mapping = {
            1: "overlap",
            2: "partial",
            3: "non_overlap",
            4: "overlap_no_annotation",
        }
        return step_folder_mapping.get(step, None)

    for author in tqdm.tqdm(os.listdir(loc)):
        cleaned_folder = os.path.join(loc, author, "cleaned")
        out_folder = os.path.join(loc, author, get_folder_name())

        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

        for file in os.listdir(cleaned_folder):
            if file.endswith("txt"):
                doc = read_book(os.path.join(cleaned_folder, file))
                char_ngram_lst = transform_doc(doc, n=3, step=step, annotate=annotate)
                joblib.dump(
                    char_ngram_lst,
                    os.path.join(out_folder, file.replace(".txt", ".pkl")),
                )


if __name__ == "__main__":
    # n = 3
    # content = 'This is a test. Nevetheless, I am testing the type-ngram method.'
    # print(LABELS)
    # print(content)
    # print(transform_doc(content, n))
    # print()
    # print(transform_doc(content, n, 2))
    # print()
    # print(transform_doc(content, n, 3))
    location = "/home/sjmaharjan/Books/author_style/data/authors"
    # load_transform_dump(loc=location,step=1)

    # load_transform_dump(loc=location,step=2)
    #
    load_transform_dump(loc=location, step=4, annotate=False)

