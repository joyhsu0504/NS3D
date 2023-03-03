#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : ccg.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import six
import copy
import inspect
import contextlib
import itertools
from collections import defaultdict
from typing import List, Optional
from jacinle.utils.printing import indent_text, print_to_string

from semantics.typing import QSTypeSystem, QSFormatContext
from semantics.syntax import CCGSyntaxType, CCGSyntaxSystem
from semantics.semantics import CCGSemantics, CCGSemanticsSugar
from semantics.composition import (
    CCGCompositionType, CCGCompositionSystem,
    CCGCompositionError, get_ccg_composition_context,
    CCGCompositionResult, CCGCoordinationImmResult
)

__all__ = ['Lexicon', 'LexiconUnion', 'CCGNode', 'compose_ccg_nodes', 'CCG']

profile = getattr(__builtins__, 'profile', lambda x: x)


class Lexicon(object):
    def __init__(self, syntax, semantics, weight=0, extra=None):
        self.syntax = syntax
        self.semantics = semantics
        self.weight = weight
        self.extra = extra

    def __str__(self):
        fmt = type(self).__name__ + '['
        fmt += 'syntax=' + str(self.syntax) + ', '
        fmt += 'semantics=' + indent_text(
            str(self.semantics.value)
        ).lstrip() + ', '
        fmt += 'weight=' + str(self.weight) + ''
        fmt += ']'
        return fmt

    __repr__ = __str__


class _LexiconUnionType(object):
    def __init__(self, *annotations):
        self.annotations = annotations


class _LexiconUnionSugar(object):
    def __getitem__(self, item):
        if isinstance(item, tuple) and not isinstance(item[0], tuple):
            return _LexiconUnionType(item)
        return _LexiconUnionType(*item)


LexiconUnion = _LexiconUnionSugar()


class CCGNode(object):
    def __init__(
        self, cs: CCGCompositionSystem,
        syntax, semantics, composition_type: CCGCompositionType,
        lexicon: Optional[Lexicon] = None,
        left: Optional['CCGNode'] = None, right: Optional['CCGNode'] = None,
        weight=None
    ):
        self.composition_system = cs
        self.syntax = syntax
        self.semantics = semantics
        self.composition_type = composition_type

        self.lexicon = lexicon
        self.left = left
        self.right = right

        self.weight = weight
        if self.weight is None:
            self.weight = self._compute_weight()

    def _compute_weight(self):
        if self.composition_type is CCGCompositionType.LEXICON:
            return self.lexicon.weight

        return self.left.weight + self.right.weight + self.composition_system.weights[self.composition_type]

    @profile
    def compose(self, right: 'CCGNode', composition_type: Optional[CCGCompositionType] = None):
        if composition_type is not None:
            try:
                ctx = get_ccg_composition_context()
                self.compose_check(right, composition_type)  # throws CCGCompositionError
                new_syntax, new_semantics = None, None
                if ctx.syntax:
                    new_syntax = self.syntax.compose(right.syntax, composition_type)
                if ctx.semantics:
                    new_semantics = self.semantics.compose(right.semantics, composition_type)
                node = self.__class__(
                    self.composition_system, new_syntax, new_semantics, composition_type,
                    left=self, right=right
                )
                return node
            except CCGCompositionError as e:
                raise e
        else:
            results = list()
            exceptions = list()

            composition_types = self.compose_guess(right)
            if composition_types is None:
                composition_types = self.composition_system.allowed_composition_types

            for composition_type in composition_types:
                try:
                    results.append((composition_type, self.compose(right, composition_type)))
                except CCGCompositionError as e:
                    exceptions.append(e)

            if len(results) == 1:
                return CCGCompositionResult(*results[0])
            elif len(results) == 0:
                with get_ccg_composition_context().exc():
                    fmt = f'Failed to compose CCGNodes {self} and {right}.\n'
                    fmt += 'Detailed messages are:\n'
                    for t, e in zip(composition_types, exceptions):
                        fmt += indent_text('Trying CCGCompositionType.{}:\n{}'.format(t.name, str(e))) + '\n'
                    raise CCGCompositionError(fmt.rstrip())
            else:
                with get_ccg_composition_context().exc():
                    fmt = f'Got ambiguous composition for CCGNodes {self} and {right}.\n'
                    fmt += 'Candidates are:\n'
                    for r in results:
                        fmt += indent_text('CCGCompositionType.' + str(r[0].name)) + '\n'
                    raise CCGCompositionError(fmt.rstrip())

    def compose_check(self, right: 'CCGNode', composition_type: CCGCompositionType):
        if (
            isinstance(self.syntax, CCGCoordinationImmResult) or
            isinstance(self.semantics, CCGCoordinationImmResult)
        ):
            raise CCGCompositionError('Can not make non-coordination composition for CCGCoordinationImmResult.')
        if (
            isinstance(self.syntax, CCGCoordinationImmResult) or
            isinstance(self.semantics, CCGCoordinationImmResult) or
            isinstance(right.syntax, CCGCoordinationImmResult) or
            isinstance(right.semantics, CCGCoordinationImmResult)
        ):
            if composition_type is not CCGCompositionType.COORDINATION:
                raise CCGCompositionError('Can not make non-coordination composition for CCGCoordinationImmResult.')

    def compose_guess(self, right: 'CCGNode') -> Optional[List[CCGCompositionType]]:
        return None

    def linearize_lexicons(self):
        if self.lexicon is not None:
            return [self.lexicon]
        return self.left.linearize_lexicons() + self.right.linearize_lexicons()

    def as_nltk_str(self):
        if self.composition_type is CCGCompositionType.LEXICON:
            if self.lexicon.extra is not None:
                meaning = str(self.lexicon.extra[0])
            else:
                with QSFormatContext(function_format_lambda=True).as_default():
                    meaning = str(self.semantics.value).replace('(', '{').replace(')', '}')
            return f'({str(self.syntax)} {meaning})'

        if self.composition_type is CCGCompositionType.COORDINATION:
            return f'({str(self.syntax)} {self.left.as_nltk_str()} {self.right.left.as_nltk_str()} {self.right.right.as_nltk_str()})'
        return f'({str(self.syntax)} {self.left.as_nltk_str()} {self.right.as_nltk_str()})'

    def format_nltk_tree(self):
        with print_to_string() as fmt:
            self.print_nltk_tree()
        return fmt.get()

    def print_nltk_tree(self):
        from nltk.tree import Tree
        parsing_nltk = Tree.fromstring(self.as_nltk_str())
        parsing_nltk.pretty_print()

    def __str__(self):
        fmt = type(self).__name__ + '[\n'
        fmt += '  syntax   : ' + str(self.syntax) + '\n'
        with QSFormatContext(function_format_lambda=True).as_default():
            fmt += '  semantics: ' + indent_text(str(self.semantics), indent_format=' ' * 13).lstrip() + '\n'
        fmt += '  weight   : ' + str(self.weight) + '\n'
        fmt += ']'
        return fmt

    __repr__ = __str__


def compose_ccg_nodes(left: CCGNode, right: CCGNode, composition_type: Optional[CCGCompositionType] = None):
    return left.compose(right, composition_type)


class CCGParsingError(Exception):
    pass


class CCG(object):
    def __init__(self, ts: QSTypeSystem, ss: CCGSyntaxSystem, cs: Optional[CCGCompositionSystem] = None):
        self.type_system = ts
        self.syntax_system = ss
        self.semantics_sugar = CCGSemanticsSugar(self.type_system)
        self.composition_system = cs
        self.lexicons = defaultdict(list)

        if self.composition_system is None:
            self.composition_system = CCGCompositionSystem.make_default()

    def clone(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        new_obj = self.__class__(self.type_system, self.syntax_system, self.composition_system)
        new_obj.lexicons = defaultdict(list)
        for k, v in self.lexicons.items():
            new_obj.lexicons[k].extend(v)
        return new_obj

    def make_node(self, arg1, arg2: Optional[CCGNode] = None, *, composition_type: Optional[CCGCompositionType] = None):
        if isinstance(arg1, six.string_types):
            if arg1 in self.lexicons:
                tot_entries = len(self.lexicons[arg1])
                if tot_entries == 1:
                    return self.make_node(self.lexicons[arg1][0])
                raise CCGParsingError('Ambiguous lexicon entry for word: "{}" (n = {}).'.format(arg1, tot_entries))
            raise CCGParsingError('Out-of-vocab word: {}.'.format(arg1))

        if isinstance(arg1, Lexicon):
            return CCGNode(self.composition_system, arg1.syntax, arg1.semantics, CCGCompositionType.LEXICON, lexicon=arg1)
        else:
            assert arg2 is not None
            return compose_ccg_nodes(arg1, arg2, composition_type=composition_type)

    @property
    def Syntax(self):
        return self.syntax_system

    @property
    def Semantics(self):
        return self.semantics_sugar

    def add_entry(self, word, lexicon):
        self.lexicons[word].append(lexicon)

    def add_entry_simple(self, word, syntax, semantics, weight=0):
        self.lexicons[word].append(Lexicon(self.Syntax[syntax], self.Semantics[semantics], weight=weight))

    def clear_entries(self, word):
        self.lexicons[word].clear()

    def update_entries(self, entries_dict):
        for word, entries in entries_dict.items():
            for entry in entries:
                self.add_entry(word, entry)

    @contextlib.contextmanager
    def define(self):
        locals_before = inspect.stack()[2][0].f_locals.copy()
        annotations_before = locals_before.get('__annotations__', dict()).copy()
        yield self
        locals_after = inspect.stack()[2][0].f_locals.copy()
        annotations_after = locals_after.get('__annotations__', dict()).copy()

        new_annotations = {
            k: v for k, v in annotations_after.items()
            if k not in annotations_before or annotations_after[k] != annotations_before[k]
        }

        if len(new_annotations) == 0:
            raise ValueError('ccg.define() is only allowed at the global scope.')

        def add_entry(word, annotation):
            assert isinstance(annotation, tuple) and len(annotation) in (2, 3)
            assert isinstance(annotation[0], CCGSyntaxType)
            assert isinstance(annotation[1], CCGSemantics)

            weight = 0
            if len(annotation) == 2:
                syntax, semantics = annotation
            else:
                syntax, semantics, weight = annotation

            self.add_entry(word, Lexicon(syntax, semantics, weight))

        for const_name, raw_annotation in new_annotations.items():
            if isinstance(raw_annotation, _LexiconUnionType):
                for a in raw_annotation.annotations:
                    add_entry(const_name, a)
            else:
                add_entry(const_name, raw_annotation)

    def parse(self, sentence, beam: Optional[int] = None, preserve_syntax_types: bool = True):
        if isinstance(sentence, six.string_types):
            sentence = sentence.split()

        length = len(sentence)
        dp = [[list() for _ in range(length + 1)] for _ in range(length)]
        for i, word in enumerate(sentence):
            if word not in self.lexicons:
                raise CCGParsingError('Out-of-vocab word: {}.'.format(word))
            dp[i][i+1] = [self.make_node(l) for l in self.lexicons[word]]

        def merge(list1, list2):
            output_list = list()
            for node1, node2 in itertools.product(list1, list2):
                try:
                    node = compose_ccg_nodes(node1, node2).result
                    output_list.append(node)
                except CCGCompositionError:
                    pass
            return output_list

        for l in range(2, length + 1):
            for i in range(0, length + 1 - l):
                j = i + l
                for k in range(i + 1, j):
                    dp[i][j].extend(merge(dp[i][k], dp[k][j]))
                if beam is not None:
                    if preserve_syntax_types:
                        dp[i][j] = filter_beam_per_type(dp[i][j], beam)
                    else:
                        dp[i][j] = sorted(dp[i][j], key=lambda x: x.weight, reverse=True)[:beam]

        return sorted(dp[0][length], key=lambda x: x.weight)

    def _format_lexicons(self):
        fmt = 'Lexicons:\n'
        # max_words_len = max([len(x) for x in self.lexicons])
        for word, lexicons in self.lexicons.items():
            for lexicon in lexicons:
                this_fmt = f'{word}: ' + str(lexicon)
                fmt += indent_text(this_fmt) + '\n'
        return fmt

    def __str__(self):
        fmt = 'Combinatory Categorial Grammar\n'
        fmt += indent_text(str(self.type_system)) + '\n'
        fmt += indent_text(str(self.syntax_system)) + '\n'
        fmt += indent_text(str(self.composition_system)) + '\n'
        fmt += indent_text(self._format_lexicons())
        return fmt

    __repr__ = __str__


def filter_beam_per_type(nodes: List[CCGNode], beam: int) -> List[CCGNode]:
    all_nodes_by_type = defaultdict(list)
    for node in nodes:
        if isinstance(node.syntax, CCGCoordinationImmResult):
            typename = f'COORDIMM({node.syntax.conj.typename}, {node.syntax.right.typename})'
        else:
            typename = node.syntax.typename
        all_nodes_by_type[typename].append(node)
    for typename, nodes in all_nodes_by_type.items():
        all_nodes_by_type[typename] = sorted(nodes, key=lambda x: x.weight, reverse=True)[:beam]
    return list(itertools.chain.from_iterable(all_nodes_by_type.values()))

