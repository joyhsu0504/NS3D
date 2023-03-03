#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : composition.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import contextlib
from typing import Dict, Union
from collections import namedtuple
from jacinle.utils.cache import cached_property
from jacinle.utils.defaults import option_context
from jacinle.utils.enum import JacEnum
from jacinle.utils.printing import indent_text

__all__ = [
    'CCGCompositionDirection', 'CCGCompositionType',
    'CCGCompositionError',
    'CCGCompositionContext', 'get_ccg_composition_context',
    'CCGCompositionResult', 'CCGCoordinationImmResult',
    'CCGComposable', 'CCGCompositionSystem'
]


class CCGCompositionDirection(JacEnum):
    LEFT = 'left'
    RIGHT = 'right'


class CCGCompositionType(JacEnum):
    LEXICON = 'lexicon'
    FORWARD_APPLICATION = 'forward_application'
    BACKWARD_APPLICATION = 'backward_application'
    COORDINATION = 'coordination'
    NONE = 'none'


class CCGCompositionError(Exception):
    pass


class CCGCompositionContext(option_context(
    '_CCGCompositionContext',
    syntax=True,
    semantics=True,
    semantics_lf=True,  # only for NeuralCCG.
    exc_verbose=True
)):
    @contextlib.contextmanager
    def exc(self, exc_type=None, from_=None):
        if self.exc_verbose:
            yield
        else:
            if exc_type is None:
                exc_type = CCGCompositionError
            if from_ is not None:
                raise exc_type() from from_
            raise exc_type()


get_ccg_composition_context = CCGCompositionContext.get_default


class CCGCompositionResult(namedtuple('_CCGCompositionResult', ['type', 'result'])):
    pass


class CCGCoordinationImmResult(namedtuple('_CCGCoordinationImmResult', ['conj', 'right'])):
    @property
    def is_none(self):
        return False

    @property
    def is_value(self):
        return False

    @property
    def is_function(self):
        return False


class CCGComposable(object):
    @property
    def is_none(self):
        return False

    @property
    def is_conj(self):
        return False

    def compose(self, right: Union['CCGComposable', CCGCoordinationImmResult], composition_type: CCGCompositionType):
        if isinstance(right, CCGCoordinationImmResult) and composition_type is not CCGCompositionType.COORDINATION:
            raise CCGCompositionError('Can not make non-coordination composition for CCGCoordinationImmResult.')
        if (self.is_none or (not isinstance(right, CCGCoordinationImmResult) and right.is_none)) and composition_type is not CCGCompositionType.NONE:
            raise CCGCompositionError('Can not make non-None composition with none elements.')

        if composition_type is CCGCompositionType.LEXICON:
            raise CCGCompositionError('Lexicon composition type is only used for leaf level nodes.')
        elif composition_type is CCGCompositionType.FORWARD_APPLICATION:
            return self.fapp(right)
        elif composition_type is CCGCompositionType.BACKWARD_APPLICATION:
            return right.bapp(self)
        elif composition_type is CCGCompositionType.COORDINATION:
            return self.coord(right)
        elif composition_type is CCGCompositionType.NONE:
            return self.none(right)

    def fapp(self, right):
        assert not self.is_none and not right.is_none
        return self._fapp(right)

    def bapp(self, left):
        assert not self.is_none and not left.is_none
        return self._bapp(left)

    def none(self, right):
        if right.is_none:
            return self
        elif self.is_none:
            return right
        with get_ccg_composition_context().exc(CCGCompositionError):
            raise CCGCompositionError('Invalid None composition: left={}, right={}.'.format(self, right))

    def coord(self, other):
        if isinstance(other, CCGCoordinationImmResult):
            return other.conj.coord3(self, other.right)
        elif self.is_conj:
            return CCGCoordinationImmResult(self, other)
        with get_ccg_composition_context().exc(CCGCompositionError):
            raise CCGCompositionError('Invalid coordination composition: left={}, right={}.'.format(self, other))

    def coord3(self, left, right):
        assert not self.is_none and not left.is_none and not right.is_none
        return self._coord3(left, right)

    def _fapp(self, right: 'CCGComposable'):
        raise NotImplementedError()

    def _bapp(self, left: 'CCGComposable'):
        raise NotImplementedError()

    def _coord3(self, left: 'CCGComposable', right: 'CCGComposable'):
        raise NotImplementedError()


class CCGCompositionSystem(object):
    def __init__(self, name, weights: Dict[CCGCompositionType, float]):
        self.name = name
        self.weights = weights

    @cached_property
    def allowed_composition_types(self):
        return [c for c in CCGCompositionType.choice_objs() if c in self.weights and c is not CCGCompositionType.LEXICON]

    @classmethod
    def make_default(cls):
        return cls('<basic>', {
            CCGCompositionType.LEXICON: 0,
            CCGCompositionType.FORWARD_APPLICATION: 0,
            CCGCompositionType.BACKWARD_APPLICATION: 0,
            CCGCompositionType.NONE: 0
        })

    @classmethod
    def make_coordination(cls):
        return cls('<coordination>', {
            CCGCompositionType.LEXICON: 0,
            CCGCompositionType.FORWARD_APPLICATION: 0,
            CCGCompositionType.BACKWARD_APPLICATION: 0,
            CCGCompositionType.COORDINATION: 0,
            CCGCompositionType.NONE: 0,
        })

    def __str__(self):
        fmt = 'Allowed composition types:\n'
        for type, weight in self.weights.items():
            fmt += '  CCGCompositionType.' + type.name + ': ' + str(weight) + '\n'
        fmt = 'CCGCompositionSystem: {}\n'.format(self.name) + indent_text(fmt.rstrip())
        return fmt

    __repr__ = __str__

    def try_compose(self, left: CCGComposable, right: CCGComposable):
        results = list()
        exceptions = list()
        for composition_type in self.allowed_composition_types:
            try:
                ret = left.compose(right, composition_type)
                results.append((composition_type, ret))
            except CCGCompositionError as e:
                exceptions.append(e)
        if len(results) == 1:
            return CCGCompositionResult(*results[0])
        elif len(results) == 0:
            with get_ccg_composition_context().exc():
                fmt = f'Failed to compose CCGNodes {self} and {right}.\n'
                fmt += 'Detailed messages are:\n'
                for t, e in zip(self.allowed_composition_types, exceptions):
                    fmt += indent_text('Trying CCGCompositionType.{}:\n{}'.format(t.name, str(e))) + '\n'
                raise CCGCompositionError(fmt.rstrip())
        else:
            with get_ccg_composition_context().exc():
                fmt = f'Got ambiguous composition for CCGNodes {self} and {right}.\n'
                fmt += 'Candidates are:\n'
                for r in results:
                    fmt += indent_text('CCGCompositionType.' + str(r[0].name)) + '\n'
                raise CCGCompositionError(fmt.rstrip())
