#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visual_reasoning_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import functools
import torch.nn as nn
from jacinle.utils.enum import JacEnum

__all__ = ['VisualReasoningFuzzyOp', 'VisualConceptEmbeddingBase', 'VisualConceptInferenceCache']


class VisualReasoningFuzzyOp(JacEnum):
    MATMUL = 'matmul'
    LOGIT = 'logit'
    MINMAX = 'minmax'


class VisualConceptEmbeddingBase(nn.Module):
    def similarity(self, query, identifier):
        raise NotImplementedError()

    def query_attribute(self, query, identifier):
        raise NotImplementedError()


class VisualConceptInferenceCache(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = dict()

    def is_cached(self, *args):
        return args in self._cache

    def get_cache(self, *args):
        return self._cache[args]

    def set_cache(self, *args, value=None):
        self._cache[args] = value
        return value

    @staticmethod
    def cached_result(cache_key):
        def wrapper(func):
            @functools.wraps(func)
            def wrapped(self, *args):
                if self.is_cached(cache_key, *args):
                    return self.get_cache(cache_key, *args)

                value = func(self, *args)
                return self.set_cache(cache_key, *args, value=value)
            return wrapped
        return wrapper