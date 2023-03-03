#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : vocab.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.


import six
import torch

import jacinle.io as io
import jaclearn.embedding.constant as const
from jacinle.utils.tqdm import tqdm

__all__ = ['Vocab', 'gen_vocab', 'gen_vocab_from_words']


class Vocab(object):
    def __init__(self, word2idx=None):
        self.word2idx = word2idx if word2idx is not None else dict()
        self._idx2word = None

    @classmethod
    def from_json(cls, json_file):
        return cls(io.load_json(json_file))

    @classmethod
    def from_dataset(cls, dataset, keys, extra_words=None):
        return gen_vocab(dataset, keys, extra_words=extra_words, cls=cls)

    @classmethod
    def from_list(cls, dataset, extra_words=None):
        return gen_vocab(dataset, extra_words=extra_words, cls=cls)

    def dump_json(self, json_file):
        io.dump_json(json_file, self.word2idx)

    def check_json_consistency(self, json_file):
        rhs = io.load_json(json_file)
        for k, v in self.word2idx.items():
            if not (k in rhs and rhs[k] == v):
                return False
        return True

    def words(self):
        return self.word2idx.keys()

    @property
    def idx2word(self):
        if self._idx2word is None or len(self.word2idx) != len(self._idx2word):
            self._idx2word = {v: k for k, v in self.word2idx.items()}
        return self._idx2word

    def __len__(self):
        return len(self.word2idx)

    def __iter__(self):
        return iter(self.word2idx.keys())

    def add(self, word):
        self.add_word(word)

    def add_word(self, word):
        self.word2idx[word] = len(self.word2idx)

    def map(self, word):
        return self.word2idx.get(
            word,
            self.word2idx.get(const.EBD_UNKNOWN, -1)
        )

    def map_sequence(self, sequence, add_be=False):
        if isinstance(sequence, six.string_types):
            sequence = sequence.split()
        sequence = [self.map(w) for w in sequence]
        if add_be:
            sequence.insert(0, self.word2idx[const.EBD_BOS])
            sequence.append(self.word2idx[const.EBD_EOS])
        return sequence

    def map_fields(self, feed_dict, fields):
        feed_dict = feed_dict.copy()
        for k in fields:
            if k in feed_dict:
                feed_dict[k] = self.map(feed_dict[k])
        return feed_dict

    def invmap_sequence(self, sequence, proc_be=False):
        if proc_be:
            raise NotImplementedError()
        if torch.is_tensor(sequence):
            sequence = sequence.detach().cpu().tolist()
        return [self.idx2word[int(x)] for x in sequence]


def gen_vocab(dataset, keys=None, extra_words=None, cls=None):
    """
    Generate a Vocabulary instance from a dataset.
    By default this function will retrieve the data using the `get_metainfo` function,
    or it will fallback to `dataset[i]` if the function does not exist.

    The function should return a dictionary. Users can specify a list of keys that will
    be returned by the `get_metainfo` function. This function will split the string indexed
    by these keys and add tokens to the vocabulary.
    If the argument `keys` is not specified, this function assumes the return of `get_metainfo`
    to be a string.

    By default, this function will add four additional tokens:
    EBD_PAD, EBD_BOS, EBD_EOS, and EBD_UNK. Users can specify additional extra tokens using the
    extra_words argument.
    """
    if cls is None:
        cls = Vocab

    all_words = set()
    for i in tqdm(len(dataset), desc='Building the vocab'):
        if hasattr(dataset, 'get_metainfo'):
            metainfo = dataset.get_metainfo(i)
        else:
            metainfo = dataset[i]

        if keys is None:
            for w in metainfo.split():
                all_words.add(w)
        else:
            for k in keys:
                if isinstance(metainfo[k], six.string_types):
                    for w in metainfo[k].split():
                        all_words.add(w)
                else:
                    for w in metainfo[k]:
                        all_words.add(w)

    vocab = cls()
    vocab.add(const.EBD_ALL_ZEROS)
    for w in sorted(all_words):
        vocab.add(w)
    for w in [const.EBD_UNKNOWN, const.EBD_BOS, const.EBD_EOS]:
        vocab.add(w)

    if extra_words is not None:
        for w in extra_words:
            vocab.add(w)

    return vocab


def gen_vocab_from_words(words, extra_words=None, cls=None):
    """Generate a Vocabulary instance from a list of words."""
    if cls is None:
        cls = Vocab
    vocab = cls()
    vocab.add(const.EBD_ALL_ZEROS)
    for w in sorted(words):
        vocab.add(w)
    for w in [const.EBD_UNKNOWN, const.EBD_BOS, const.EBD_EOS]:
        vocab.add(w)
    if extra_words is not None:
        for w in extra_words:
            vocab.add(w)
    return vocab


