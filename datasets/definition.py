#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : definition.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

__all__ = ['DatasetDefinitionBase', 'get_global_definition', 'set_global_definition', 'gdef']


class DatasetDefinitionBase(object):
    pass


class GlobalDefinitionWrapper(object):
    def __getattr__(self, item):
        return getattr(get_global_definition(), item)

    def __setattr__(self, key, value):
        raise AttributeError('Cannot set the attr of `gdef`.')


gdef = GlobalDefinitionWrapper()


_GLOBAL_DEF = None


def get_global_definition():
    global _GLOBAL_DEF
    assert _GLOBAL_DEF is not None
    return _GLOBAL_DEF


def set_global_definition(def_):
    global _GLOBAL_DEF
    assert _GLOBAL_DEF is None
    _GLOBAL_DEF = def_


