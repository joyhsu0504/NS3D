#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : semantics.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

from collections import namedtuple
from jacinle.utils.cache import cached_property
from jacinle.utils.printing import indent_text
from semantics.typing import (
    is_function, is_value, get_type, QSObject, QSFunctionObject, QSFunction, QSFunctionApplication,
    QSConstantType, QSVariableType,
    QSFunctionType, QSOverloadedFunctionType,
    QSFunctionArgumentResolutionError, QSFunctionArgumentResolutionContext,
    AnonymousArgument
)
from .composition import CCGCompositionType, CCGComposable, CCGCompositionError, get_ccg_composition_context

__all__ = [
    'CCGSemanticsCompositionError',
    'CCGSemanticsConjValue', 'CCGSemanticsBasicConjValue',
    'CCGSemanticsLazyValue',
    'CCGSemantics', 'CCGSemanticsSugar'
]


class CCGSemanticsCompositionError(CCGCompositionError):
    def __init__(self, message=None, left=None, right=None, error=None, conj=None):
        if message is None:
            super().__init__(None)
        else:
            message += '\nLeft:' + indent_text(str(left)) + '\nRight:' + indent_text(str(right))
            if conj is not None:
                message += '\nConj: ' + indent_text(str(conj))
            message += '\nOriginal error message: ' + indent_text(str(error)).lstrip()
            super().__init__('(Semantics) ' + message)


class CCGSemantics(CCGComposable):
    def __init__(self, value, *, is_conj=False):
        """
        Possible value types are:

            - None
            - PythonFunction
            - CCGSemanticsLazyValue
            - QSFunction
            - QSObject
            - QSFunctionApplication

        """
        self.value = value
        self._is_conj = is_conj

    @property
    def is_conj(self):
        return self._is_conj

    @property
    def is_none(self):
        return self.value is None

    @property
    def is_py_function(self):
        return not self.is_function and callable(self.value)

    @property
    def is_lazy(self):
        return isinstance(self.value, CCGSemanticsLazyValue)

    @property
    def is_function(self):
        if self.is_lazy:
            raise ValueError('Cannot check is_function for CCGSemanticsLazyValue')
        return is_function(self.value)

    @property
    def is_value(self):
        if self.is_lazy:
            raise ValueError('Cannot check is_value for CCGSemanticsLazyValue')
        return is_value(self.value)

    @property
    def value_executed(self):
        if self.is_lazy:
            return self.value.execute()
        return self.value

    @property
    def type(self):
        if self.is_value:
            return get_type(self.value)
        elif self.is_function:
            return self.value.type
        else:
            raise AttributeError('Cannot get the type of None, PyFunction, or Lazy semantics.')

    @property
    def return_type(self):
        if self.is_value:
            return get_type(self.value)
        elif self.is_function:
            return self.value.type.return_type
        else:
            raise AttributeError('Cannot get the return type of None, PyFunction, or Lazy semantics.')

    @cached_property
    def arity(self):
        if self.is_value:
            return 0
        elif self.is_function:
            return self.value.type.nr_variable_arguments
        else:
            raise AttributeError('Cannot get the arity of None, PyFunction, or Lazy semantics.')

    def __str__(self):
        if self.value is None:
            return type(self).__name__ + '[None]'
        if self.is_conj:
            return type(self).__name__ + '[' + str(self.value) + ', CONJ]'
        return type(self).__name__ + '[' + str(self.value) + ']'

    __repr__ = __str__

    @cached_property
    def hash(self):  # for set/dict indexing.
        return str(self)

    def _fapp(self, right: 'CCGSemantics'):
        return type(self)(_forward_application(self.value, right.value))

    def _bapp(self, left: 'CCGSemantics'):
        return type(self)(_backward_application(left.value, self.value))

    def _coord3(self, left: 'CCGSemantics', right: 'CCGSemantics'):
        return type(left)(_coordination(left.value, self.value, right.value))

    def canonize_parameters(self):
        """Return a new CCGSemantics object with argument reordered: functions, variables, constants."""
        assert not self.is_none
        if isinstance(self.value, (QSObject, QSFunctionApplication)):
            return self

        f = self.value
        if f.overridden_call is None:
            return self

        assert not isinstance(f.type, QSOverloadedFunctionType)

        function_call = f.overridden_call

        if len(f.type.argument_types) == 0:
            return self.__class__(function_call())

        args = [
            QSFunctionObject(t, QSFunction(t, name=AnonymousArgument(i))) if isinstance(t, QSFunctionType) else QSObject(t, AnonymousArgument(i))
            for i, t in enumerate(f.type.argument_types)
        ]
        ret = function_call(*args)

        assert isinstance(ret, QSFunctionApplication)

        function_args, variable_args, constant_args = set(), set(), set()  # Python sets are ordered.

        def walk(node: QSFunctionApplication):
            if isinstance(node.function.name, AnonymousArgument):
                function_args.add(node.function.name.name)

            for arg in node.args:
                if isinstance(arg, QSObject):
                    if isinstance(arg.value, AnonymousArgument):
                        if isinstance(arg.type, QSConstantType):
                            constant_args.add(arg.value.name)
                        elif isinstance(arg.type, QSVariableType):
                            variable_args.add(arg.value.name)
                        else:
                            raise TypeError('Unknown type for anonymous argument #{}, type = {}.'.format(
                                arg.value.name, arg.type
                            ))
                elif isinstance(arg, QSFunctionApplication):
                    walk(arg)
                else:
                    raise TypeError('Unknown type for anonymous argument type {}.'.format(type(arg)))

        try:
            walk(ret)
        finally:
            del walk

        new_argument_mapping = list(function_args) + list(variable_args) + list(constant_args)
        return self.__class__(f.remap_arguments(new_argument_mapping))


def _forward_application(left, right):
    ctx = get_ccg_composition_context()

    if is_function(left):
        try:
            with QSFunctionArgumentResolutionContext(exc_verbose=ctx.exc_verbose).as_default():
                return left.partial(right)
        except QSFunctionArgumentResolutionError as e:
            with ctx.exc(CCGSemanticsCompositionError, e):
                raise CCGSemanticsCompositionError(f'Cannot make forward application.', left, right, e) from e
    with ctx.exc(CCGSemanticsCompositionError):
        raise CCGSemanticsCompositionError(f'Cannot make forward application.', left, right, 'Functor/Value types do not match.')


def _backward_application(left, right):
    ctx = get_ccg_composition_context()

    if is_function(right):
        try:
            with QSFunctionArgumentResolutionContext(exc_verbose=ctx.exc_verbose).as_default():
                return right.partial(left)
        except QSFunctionArgumentResolutionError as e:
            with ctx.exc(CCGSemanticsCompositionError, e):
                raise CCGSemanticsCompositionError(f'Cannot make backward application.', left, right, e) from e
    with ctx.exc(CCGSemanticsCompositionError):
        raise CCGSemanticsCompositionError(f'Cannot make backward application.', left, right, 'Functor/Value types do not match.')


def _coordination(left, conj, right):
    ctx = get_ccg_composition_context()

    if is_function(left) and is_function(right) and callable(conj):
        if left.type.nr_variable_arguments == right.type.nr_variable_arguments:
            return conj(left, right)
    if is_value(left) and is_value(right) and callable(conj):
        return conj(left, right)

    with ctx.exc(CCGSemanticsCompositionError):
        raise CCGSemanticsCompositionError(
            f'Cannot make coordination.',
            left, right, conj=conj,
            error='Functor arity does not match.'
        )


class CCGSemanticsConjValue(object):
    def __init__(self, impl):
        self.impl = impl

    def __call__(self, left, right):
        return self.impl(left, right)


class CCGSemanticsBasicConjValue(CCGSemanticsConjValue):
    def __call__(self, left, right):
        if is_function(left) and is_function(right):
            def body(*args, **kwargs):
                return self.impl(left(*args, **kwargs), right(*args, **kwargs))
            return QSFunction(left.type, overridden_call=body, name='<conj>')
        if is_value(left) and is_value(right):
            return self.impl(left, right)


class CCGSemanticsLazyValue(namedtuple(
    '_NeuralCCGLazySemanticsValue',
    ('composition_type', 'left', 'right', 'conj'),
    defaults=[None, None, None, None]
)):
    def execute(self):
        left, right = self.left, self.right
        if isinstance(left, CCGSemanticsLazyValue):
            left = left.execute()
        if isinstance(right, CCGSemanticsLazyValue):
            right = right.execute()

        if self.composition_type in (CCGCompositionType.FORWARD_APPLICATION, CCGCompositionType.BACKWARD_APPLICATION):
            if self.composition_type is CCGCompositionType.FORWARD_APPLICATION:
                return _forward_application(left, right)
            else:
                return _backward_application(left, right)
        elif self.composition_type is CCGCompositionType.COORDINATION:

            return _coordination(left, self.conj, right)
        else:
            raise NotImplementedError('Unimplemented lazy composition type: {}.'.format(self.composition_type))


class CCGSemanticsSugar(object):
    def __init__(self, ts):
        self.type_system = ts

    def __getitem__(self, item):
        if item is None:
            return CCGSemantics(None)
        if isinstance(item, CCGSemantics):
            return item
        if isinstance(item, (QSFunction, QSObject, QSFunctionApplication)):
            return CCGSemantics(item)
        if isinstance(item, tuple):
            assert len(item) == 2
            return CCGSemantics(self.type_system.lam(item[0], typing_cues=item[1]))
        assert callable(item)
        return CCGSemantics(self.type_system.lam(item))
