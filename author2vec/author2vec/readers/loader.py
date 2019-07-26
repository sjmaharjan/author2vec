

def read_book(filename, encoding='utf-8'):
    with open(filename, mode='r', encoding=encoding) as f_in:
        content = " ".join([line.replace('\r\n', '') for line in f_in.readlines()])
    return content if content else ''

