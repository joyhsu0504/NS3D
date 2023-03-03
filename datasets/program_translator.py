#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : program_translator.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

from collections import defaultdict
from copy import deepcopy

import semantics.typing as T
from datasets.definition import gdef

__all__ = ['nsclseq_to_nscltree', 'nscltree_to_nsclseq', 'nscltree_to_nsclv2', 'nsclseq_to_nsclqsseq', 'nscltree_to_nsclqstree', 'iter_nscltree']


def nsclseq_to_nscltree(seq_program):
    def dfs(sblock):
        tblock = deepcopy(sblock)
        input_ids = tblock.pop('inputs')
        tblock['inputs'] = [dfs(seq_program[i]) for i in input_ids]
        return tblock

    try:
        return dfs(seq_program[-1])
    finally:
        del dfs


def nscltree_to_nsclseq(tree_program):
    tree_program = deepcopy(tree_program)
    seq_program = list()

    def dfs(tblock):
        sblock = tblock.copy()
        input_blocks = sblock.pop('inputs')
        sblock['inputs'] = [dfs(b) for b in input_blocks]
        seq_program.append(sblock)
        return len(seq_program) - 1

    try:
        dfs(tree_program)
        return seq_program
    finally:
        del dfs


def nscltree_to_nsclv2(tree_program, expand_filter=True):
    tree_program = deepcopy(tree_program)

    def dfs(tblock):
        op = tblock['op']
        if expand_filter and op == 'filter':
            cur = dfs(tblock['inputs'][0])
            for c in tblock['concept']:
                cur = gdef.type_system.f_filter(cur, T.QSObject(gdef.type_system.t_concept, c))
            return cur

        functype = gdef.type_system.functions[op].type
        inputs = [dfs(i) for i in tblock['inputs']]
        for argtype in functype.argument_types[len(inputs):]:
            inputs.append(T.QSObject(argtype, tblock[argtype.typename]))
        return gdef.type_system.functions[op](*inputs)

    try:
        return dfs(tree_program)
    finally:
        del dfs


def nsclseq_to_nsclqsseq(seq_program):
    qs_seq = deepcopy(seq_program)
    cached = defaultdict(list)

    for sblock in qs_seq:
        for param_type in gdef.type_system.types:
            if isinstance(param_type, T.QSConstantType):
                param_type = param_type.typename
                if param_type in sblock:
                    sblock[param_type + '_idx'] = len(cached[param_type])
                    sblock[param_type + '_values'] = cached[param_type]
                    cached[param_type].append(sblock[param_type])

    return qs_seq


def nscltree_to_nsclqstree(tree_program):
    qs_tree = deepcopy(tree_program)
    cached = defaultdict(list)

    for tblock in iter_nscltree(qs_tree):
        for param_type in gdef.type_system.types:
            if isinstance(param_type, T.QSConstantType):
                param_type = param_type.typename
                if param_type in tblock:
                    tblock[param_type + '_idx'] = len(cached[param_type])
                    tblock[param_type + '_values'] = cached[param_type]
                    cached[param_type].append(tblock[param_type])

    return qs_tree


def iter_nscltree(tree_program):
    yield tree_program
    for i in tree_program['inputs']:
        yield from iter_nscltree(i)