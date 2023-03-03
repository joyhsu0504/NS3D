#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : syntax.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

from typing import List, Union, Tuple
from .composition import CCGCompositionDirection, CCGCompositionError, get_ccg_composition_context, CCGComposable
from jacinle.utils.cache import cached_property
from jacinle.utils.printing import indent_text

__all__ = [
    'CCGSyntaxCompositionError', 'CCGSyntaxTypeParsingError',
    'CCGSyntaxType', 'CCGBasicSyntaxType', 'CCGConjSyntaxType', 'CCGComposedSyntaxType',
    'CCGSyntaxSystem'
]


class CCGSyntaxCompositionError(CCGCompositionError):
    def __init__(self, message=None):
        if message is None:
            super().__init__(None)
        else:
            super().__init__('(Syntax) ' + message)


class CCGSyntaxTypeParsingError(Exception):
    pass


class CCGSyntaxType(CCGComposable):
    def __init__(self, typename=None):
        self.typename = typename

    @property
    def parenthesis_typename(self):
        return self.typename

    @property
    def arity(self):
        return 0

    @property
    def is_none(self):
        return self.typename is None

    def _fapp(self, right: 'CCGSyntaxType'):
        return _forward_application(self, right)

    def _bapp(self, left: 'CCGSyntaxType'):
        return _backward_application(left, self)

    def _coord3(self, left: 'CCGSyntaxType', right: 'CCGSyntaxType'):
        return _coordination(left, self, right)

    def __str__(self):
        return str(self.typename)

    __repr__ = __str__

    def __truediv__(self, other):
        return CCGComposedSyntaxType(self, other, direction=CCGCompositionDirection.RIGHT)

    def __floordiv__(self, other):
        return CCGComposedSyntaxType(self, other, direction=CCGCompositionDirection.LEFT)

    def __eq__(self, other):
        return self.typename == other.typename

    def __ne__(self, other):
        return self.typename != other.typename

    def __hash__(self):
        return str(self)

    def __lt__(self, other):
        """Customized comparison function for sorting a list of syntax types."""
        a, b = str(self), str(other)
        return (a.count('/') + a.count('\\'), a) < (b.count('/') + b.count('\\'), b)

    def flatten(self) -> List[Union['CCGSyntaxType', Tuple['CCGSyntaxType', CCGCompositionDirection]]]:
        raise NotImplementedError()


class CCGBasicSyntaxType(CCGSyntaxType):
    def flatten(self):
        return [self]


class CCGConjSyntaxType(CCGSyntaxType):
    @property
    def is_conj(self):
        return True

    def __call__(self, left: CCGSyntaxType, right: CCGSyntaxType):
        return left

    def flatten(self):
        return [self]


class CCGComposedSyntaxType(CCGSyntaxType):
    def __init__(self, main: CCGSyntaxType, sub: CCGSyntaxType, direction: CCGCompositionDirection):
        self.main = main
        self.sub = sub
        self.direction = CCGCompositionDirection.from_string(direction)

        if self.direction is CCGCompositionDirection.RIGHT:
            typename = self.main.typename + '/' + self.sub.parenthesis_typename
        else:
            typename = self.main.typename + '\\' + self.sub.parenthesis_typename
        super().__init__(typename)

    @cached_property
    def arity(self):
        return self.main.arity + 1

    @property
    def parenthesis_typename(self):
        return '{' + f'{self.typename}' + '}'

    def flatten(self):
        ret = self.main.flatten()
        ret.append((self.sub, self.direction))
        return ret


def _forward_application(left, right):
    if isinstance(left, CCGComposedSyntaxType):
        if left.direction == CCGCompositionDirection.RIGHT:
            if left.sub == right:
                return left.main
    with get_ccg_composition_context().exc(CCGSyntaxCompositionError):
        raise CCGSyntaxCompositionError(f'Cannot make forward application of {left} and {right}.')


def _backward_application(left, right):
    if isinstance(right, CCGComposedSyntaxType):
        if right.direction == CCGCompositionDirection.LEFT:
            if right.sub == left:
                return right.main
    with get_ccg_composition_context().exc(CCGSyntaxCompositionError):
        raise CCGSyntaxCompositionError(f'Cannot make backward application of {left} and {right}.')


def _coordination(left, conj, right):
    if left == right and isinstance(conj, CCGConjSyntaxType):
        return conj(left, right)
    with get_ccg_composition_context().exc(CCGSyntaxCompositionError):
        raise CCGSyntaxCompositionError(f'Cannot make coordination of {left} {conj} {right}.')


class CCGSyntaxSystem(object):
    def __init__(self, name='<CCGSyntaxSystem>'):
        self.name = name
        self.types = dict()

    def define_basic_type(self, type):
        if isinstance(type, CCGSyntaxType):
            self.types[type.typename] = type
        else:
            self.types[type] = CCGBasicSyntaxType(type)

    def define_conj_type(self, type):
        if isinstance(type, CCGSyntaxType):
            self.types[type.typename] = type
        else:
            self.types[type] = CCGConjSyntaxType(type)

    def parse(self, string):
        def parse_inner(current):
            if current == '':
                raise CCGSyntaxTypeParsingError('Invalid syntax type string (got empty type): {}.'.format(string))
            nr_parenthesis = 0
            last_op = None
            for i, c in enumerate(current):
                if c in r'\/':
                    if nr_parenthesis == 0:
                        last_op = i
                if c == '(':
                    nr_parenthesis += 1
                elif c == ')':
                    nr_parenthesis -= 1
                    if nr_parenthesis < 0:
                        raise CCGSyntaxTypeParsingError('Invalid parenthesis (extra ")"): {}.'.format(string))
            if nr_parenthesis != 0:
                raise CCGSyntaxTypeParsingError('Invalid parenthesis (extra "("): {}.'.format(string))

            if last_op is None:
                if current[0] == '(' and current[-1] == ')':
                    return parse_inner(current[1:-1])
                else:
                    if current in self.types:
                        return self.types[current]
                    else:
                        raise CCGSyntaxTypeParsingError(
                            'Unknown basic syntax type {} during parsing {}.'.format(current, string)
                        )

            last_op_value = CCGCompositionDirection.RIGHT if current[last_op] == '/' else CCGCompositionDirection.LEFT
            return CCGComposedSyntaxType(
                parse_inner(current[:last_op]), parse_inner(current[last_op+1:]),
                direction=last_op_value
            )

        try:
            return parse_inner(string)
        finally:
            del parse_inner

    def __getitem__(self, item):
        if item is None:
            return CCGSyntaxType(None)
        if isinstance(item, CCGSyntaxType):
            return item
        return self.parse(item)

    def __str__(self):
        fmt = 'Basic types:\n'
        for type in self.types.values():
            fmt += '  ' + str(type) + '\n'
        fmt = 'CCGSyntaxSystem: {}\n'.format(self.name) + indent_text(fmt.rstrip())
        return fmt

    __repr__ = __str__