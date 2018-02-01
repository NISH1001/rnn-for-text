#!/usr/bin/env python3

from collections import Counter

import numpy as np
import re


tokens = {
    '.': '||period||',
    ',': '||comma||',
   '"': '||quotation_mark||',
   ';': '||semicolon||',
   '!': '||exclamation_mark||',
   '?': '||question_mark||',
   '(': '||left_parentheses||',
   ')': '||right_parentheses||',
   '--': '||dash||',
   '\n': '||return||'
}

def reduce_whitespaces(text):
    return re.sub(r"[\s]+", " ", text)

def create_lookup_tables(text):
    counts = Counter(text)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = { word:i for i, word in enumerate(vocab, 1)  }
    int_to_vocab = { i:word for i, word in enumerate(vocab, 1)  }
    return vocab_to_int, int_to_vocab

def process_text(text):
    data = reduce_whitespaces(text)
    # tokenize every word
    for key, token in tokens.items():
        data = data.replace(key, ' {} '.format(token))
    data = data.lower()

    data = data.split()
    vocab_to_int, int_to_vocab = create_lookup_tables(data)
    int_text = [ vocab_to_int[word] for word in data ]
    return int_text, vocab_to_int, int_to_vocab


def main():
    text = "hello i am paradox. i am gru"
    int_text, vocab_to_int, int_to_vocab = process_text(text)
    print(text)
    print(vocab_to_int)
    print(int_text)

if __name__ == "__main__":
    main()

