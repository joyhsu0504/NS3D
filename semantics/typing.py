#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : typing.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Typing system for semantics.
"""

import six
import collections
import itertools
import contextlib
import inspect
import re
from typing import Union, List, Optional, Callable
from jacinle.logging import get_logger
from jacinle.utils.cache import cached_property
from jacinle.utils.defaults import option_context
from jacinle.utils.printing import indent_text

__all__ = [
    'QSTypeConversionError',
    'is_function', 'is_value', 'get_type', 'get_types',
    'QSFormatContext', 'get_format_context',
    'QSType', 'QSConstantType', 'QSVariableType', 'QSVariableSetType', 'QSVariableUnionType',
    'QSFunctionType', 'QSOverloadedFunctionType',
    'QSOverloadedFunctionAmbiguousResolutions',
    'QSFunctionArgumentResolutionContext', 'get_function_argument_resolution_context',
    'QSFunctionArgumentUnset', 'QSFunctionArgumentResolutionError',
    'QSTypingUnionType', 'QSTypingUnion',
    'QSObject', 'QSConstant', 'QSVariable', 'QSVariableSet', 'QSVariableUnion', 'QSFunctionObject',
    'QSFunction', 'QSFunctionApplication',
    'AnonymousArgument', 'AnonymousArgumentGen',
    'QSTypingFunction',
    'QSTypeSystem'
]

logger = get_logger(__file__)


class QSTypeConversionError(Exception):
    pass


def is_function(value):
    return isinstance(value, QSFunction)


def is_value(value):
    return isinstance(value, (QSObject, QSFunctionApplication))


def get_type(value):
    if value is QSFunctionArgumentUnset:
        return QSFunctionArgumentUnset
    if isinstance(value, QSFunction):
        return value.type
    if isinstance(value, QSObject):
        return value.type
    if isinstance(value, QSFunctionApplication):
        return value.return_type

    raise ValueError('Unknown value type: {} (type = {}).'.format(value, type(value)))


def get_types(args=None, kwargs=None):
    ret = list()
    if args is not None:
        ret.append(tuple(get_type(v) for v in args))
    if kwargs is not None:
        ret.append({k: get_type(v) for k, v in kwargs.items()})
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


class QSFormatContext(option_context(
    '_QSFunctionArgumentResolutionContext',
    type_format_cls=False, object_format_type=True, function_format_lambda=False, expr_max_length=120
)):
    pass


get_format_context = QSFormatContext.get_default


class QSType(object):
    def __init__(self, typename):
        self.typename = typename

    def __str__(self):
        if get_format_context().type_format_cls:
            return self.long_str()
        else:
            return self.short_str()

    __repr__ = __str__

    def short_str(self):
        return self.typename if self.typename is not None else str(self)

    def long_str(self):
        return f'QSType[{self.typename}]'

    def __call__(self, value):
        return QSObject(self, value)

    def __instancecheck__(self, instance):
        instance_type = get_type(instance)
        return self.downcast_compatible(instance_type)

    def downcast_compatible(self, other):
        return self.typename == other.typename or other == AnyType


AnyType = QSType('AnyType')


class QSConstantType(QSType):
    def long_str(self):
        return f'QSConst[{self.typename}]'

    def __call__(self, value):
        return QSConstant(self, value)


class QSVariableType(QSType):
    def long_str(self):
        return f'QSValue[{self.typename}]'

    def __call__(self, value):
        return QSVariable(self, value)


class QSVariableSetType(QSVariableType):
    def __init__(self, base_type, typename=None):
        if typename is None:
            typename = base_type.typename + '_set'

        super().__init__(typename)
        self.base_type = base_type

    def long_str(self):
        return f'QSValueSet[{self.typename}]'

    def __call__(self, value):
        return QSVariableSet(self, value)


class QSVariableUnionType(QSVariableType):
    def __init__(self, *types, denotations=None, typename=None):
        if typename is None:
            typename = '_'.join([t.typename for t in types]) + '_union'

        super().__init__(typename)
        self.types = tuple(types)
        self.denotations = denotations

        if self.denotations is not None:
            assert isinstance(self.denotations, (tuple, list))
            self.denotations = type(self.denotations)

    def long_str(self):
        return f'QSValueUnion[{self.typename}]'

    def __call__(self, *values):
        return QSVariableUnion(self, values)


class QSFunctionArgumentResolutionError(Exception):
    pass


QSFunctionArgumentUnset = object()


class QSFunctionArgumentResolutionContext(option_context(
    '_QSFunctionArgumentResolutionContext',
    check_missing=True, check_type=True, check_overloaded_ambiguity=True, exc_verbose=True
)):
    @contextlib.contextmanager
    def exc(self, exc_type=None, from_=None):
        if self.exc_verbose:
            yield
        else:
            if exc_type is None:
                exc_type = QSFunctionArgumentResolutionError
            if from_ is not None:
                raise exc_type() from from_
            raise exc_type()


get_function_argument_resolution_context = QSFunctionArgumentResolutionContext.get_default


class QSFunctionType(QSType):
    def __init__(self, argument_types, return_type, argument_denotations=None):
        super().__init__(None)
        self.argument_denotations = argument_denotations
        self.argument_types = argument_types
        self.return_type = return_type

        if self.argument_denotations is not None:
            assert len(self.argument_denotations) == len(self.argument_types)

        self.typename = self._gen_typename()

    def _gen_typename(self):
        return '{' + ','.join([x.typename for x in self.argument_types]) + '}->' + self.return_type.typename

    @property
    def nr_arguments(self):
        return len(self.argument_types)

    @cached_property
    def nr_variable_arguments(self):
        return len(list(filter(lambda x: isinstance(x, QSVariableType), self.argument_types)))

    @cached_property
    def argument_denotation2index(self):
        assert self.argument_denotations is not None
        return {v: k for k, v in enumerate(self.argument_denotations)}

    @classmethod
    def from_annotation(cls, function, sig=None):
        if sig is None:
            sig = inspect.signature(function)
        argument_types = list()
        argument_denotations = list()
        for i, (name, param) in enumerate(sig.parameters.items()):
            if i == 0 and name == 'self':
                continue  # is an instancemethod.
            if i == 0 and name == 'self':
                continue  # is a classmethod.
            argument_types.append(param.annotation)
            argument_denotations.append(name)
        return_type = sig.return_annotation

        if inspect._empty in argument_types or return_type is inspect._empty:
            raise QSFunctionArgumentResolutionError('Incomplete argument and return type annotation for {}.'.format(function))

        function_type = cls(
            argument_types, return_type, argument_denotations=argument_denotations
        )

        if QSOverloadedFunctionType.has_union_arguments(function_type):
            return QSOverloadedFunctionType.from_union_arguments(function_type)

        return function_type

    def resolve_args(self, *args, **kwargs):
        resolution_context = get_function_argument_resolution_context()
        denotation2index = {f'#{i}': i for i in range(self.nr_arguments)}
        if self.argument_denotations is None:
            if len(kwargs) > 0:
                if all(map(lambda x: x.startswith('#'), kwargs)):
                    pass
                else:
                    raise ValueError('Keyword arguments must be used with argument denotations.')
        else:
            denotation2index.update(self.argument_denotation2index)

        arguments = [QSFunctionArgumentUnset for _ in range(self.nr_arguments)]
        if len(args) + len(kwargs) > self.nr_arguments:
            with resolution_context.exc():
                raise QSFunctionArgumentResolutionError(
                    f'Function {self} takes {len(self.argument_types)} arguments, got {len(args) + len(kwargs)}.'
                )
        for i in range(len(args)):
            arguments[i] = args[i]
        for k, v in kwargs.items():
            if k not in denotation2index:
                with resolution_context.exc():
                    raise QSFunctionArgumentResolutionError(
                        'Got unknown keyword argument: {} when invoking function {}.'.format(k, str(self))
                    )
            i = denotation2index[k]
            if arguments[i] is not QSFunctionArgumentUnset:
                with resolution_context.exc():
                    raise QSFunctionArgumentResolutionError(
                        'Got duplicated argument for keyword argument: {} when invoking function {}.'.format(k, str(self))
                    )
            arguments[i] = v

        if resolution_context.check_missing:
            for i in range(self.nr_arguments):
                if arguments[i] is QSFunctionArgumentUnset:
                    with resolution_context.exc():
                        raise QSFunctionArgumentResolutionError(
                            'Missing argument #{} when invoking function {}.'.format(i, str(self))
                        )

        if resolution_context.check_type:
            arguments_types = get_types(arguments)
            for i in range(self.nr_arguments):
                if (arguments_types[i] is not QSFunctionArgumentUnset and
                    not self.argument_types[i].downcast_compatible(arguments_types[i])
                ):
                    with resolution_context.exc():
                        raise QSFunctionArgumentResolutionError(
                            'Typecheck failed for argument #{} while invoking the function {}.\nInvoked with types: {}.'.format(
                            i, str(self), arguments_types
                        ))

        return arguments

    def eq_arguments(self, other: 'QSFunctionType'):
        for t1, t2 in zip(self.argument_types, other.argument_types):
            if t1 != t2:
                return False
        return True

    def __str__(self):
        if self.argument_denotations is None:
            fmt = '({}) -> {}'.format(
                ', '.join([x.short_str() for x in self.argument_types]),
                self.return_type.short_str()
            )
        else:
            fmt = '({}) -> {}'.format(
                ', '.join([d + ': ' + x.short_str() for x, d in zip(self.argument_types, self.argument_denotations)]),
                self.return_type.short_str()
            )
        return fmt

    __repr__ = __str__


class QSOverloadedFunctionType(QSType):
    def __init__(self, types):
        super().__init__(None)

        types_flatten = list()
        for type in types:
            if isinstance(type, QSOverloadedFunctionType):
                types_flatten.extend(type.types)
            else:
                assert isinstance(type, QSFunctionType)
                types_flatten.append(type)

        self.types = types_flatten
        self.typename = self._gen_typename()

    def _gen_typename(self):
        return 'Union{' + ','.join([x.typename for x in self.types]) + '}'

    @property
    def nr_types(self):
        return len(self.types)

    @classmethod
    def has_union_arguments(cls, function_type: QSFunctionType):
        for arg_type in function_type.argument_types:
            if isinstance(arg_type, QSTypingUnionType):
                return True
        return False

    @classmethod
    def from_union_arguments(cls, function_type):
        product_bases = list()
        for arg_type in function_type.argument_types:
            if isinstance(arg_type, QSTypingUnionType):
                product_bases.append(arg_type.types)
            else:
                product_bases.append([arg_type])

        product_types = tuple(
            QSFunctionType(
                arg_type, function_type.return_type,
                argument_denotations=function_type.argument_denotations
            ) for arg_type in itertools.product(*product_bases)
        )
        return cls(product_types, typename=function_type.typename)

    def resolve_type_and_args(self, *args, **kwargs):
        resolution_context = get_function_argument_resolution_context()

        success_results = list()
        exceptions = list()
        for i, type in enumerate(self.types):
            assert isinstance(type, QSFunctionType)
            try:
                arguments = type.resolve_args(*args, **kwargs)
                success_results.append((i, type, arguments))
            except QSFunctionArgumentResolutionError as e:
                exceptions.append(e)

        if len(success_results) == 1:
            return success_results[0]
        elif len(success_results) == 0:
            with resolution_context.exc():
                fmt = 'Failed to resolve overloaded function{}.\n'.format('' if self.typename is None else ' ' + self.typename)
                fmt += 'Detailed messages are:\n'
                for type, r in zip(self.types, exceptions):
                    this_fmt = 'Trying ' + str(type) + ':\n'
                    this_fmt += indent_text(str(r))
                    fmt += indent_text(this_fmt) + '\n'
                raise QSFunctionArgumentResolutionError(fmt.rstrip())
        else:
            if resolution_context.check_overloaded_ambiguity:
                with resolution_context.exc():
                    fmt = 'Got ambiguous application of overloaded function{}.\n'.format('' if self.typename is None else ' ' + self.typename)
                    fmt += 'Candidates are:\n'
                    for r in success_results:
                        fmt += indent_text(str(r[1])) + '\n'
                    fmt += 'Invoked with arguments: {}.'.format(str(success_results[0][2]))
                    raise QSFunctionArgumentResolutionError(fmt)
            else:
                return QSOverloadedFunctionAmbiguousResolutions(success_results)

    def __instancecheck__(self, instance):
        for t in self.types:
            if isinstance(instance, t):
                return True
        return False

    def __str__(self):
        return 'OverloadedFunction[\n  {}\n]'.format('\n  '.join(str(x) for x in self.types))

    __repr__ = __str__


class QSOverloadedFunctionAmbiguousResolutions(list):
    pass


class QSTypingUnionType(QSType):
    def __init__(self, types):
        super().__init__(None)
        self.types = types

    def __str__(self):
        return '{}[{}]'.format('TypeUnion', ', '.join([str(x) for x in self.types]))

    __repr__ = __str__

    def __instancecheck__(self, instance):
        for t in self.types:
            if isinstance(instance, t):
                return True
        return False


class _QSTypingUnionSugar(object):
    def __getitem__(self, item):
        return QSTypingUnionType(item)


QSTypingUnion = _QSTypingUnionSugar()


class QSObject(object):
    def __init__(self, type: QSType, value):
        self.type = type
        self.value = value
        self._check_type()

    def _check_type(self):
        pass

    def __str__(self):
        if get_format_context().object_format_type:
            return f'{self.value}: {self.type}'
        else:
            return self.value

    __repr__ = __str__


class QSConstant(QSObject):
    def _check_type(self):
        assert isinstance(self.type, QSConstantType)


class _QSFunctionPlaceholderApplication(QSObject):
    def __init__(self, rtype, name, arguments):
        self.name = name
        self.arguments = arguments
        super().__init__(rtype, str(self))

    def __str__(self):
        return self.name + '(' + ', '.join([str(x) for x in self.arguments]) + ')'

    __repr__ = __str__


class _QSFunctionPlaceholderValue(object):
    def __init__(self, name, rtype):
        self.name = name
        self.rtype = rtype

    def __call__(self, *arguments):
        return _QSFunctionPlaceholderApplication(self.rtype, self.name, arguments)

    def __str__(self):
        return self.name

    __repr__ = __str__


class QSFunctionObject(QSObject):
    def __call__(self, *arguments):
        return self.value(*arguments)

    def __str__(self):
        if isinstance(self.value, QSFunction):
            if get_format_context().object_format_type:
                return f'{self.value.name}: {self.type}'
            else:
                return self.value.name
        else:
            return super().__str__(self)

    __repr__ = __str__



class QSVariable(QSObject):
    def _check_type(self):
        assert isinstance(self.type, QSVariableType)


class QSVariableSet(QSVariable):
    def _check_type(self):
        assert isinstance(self.type, QSVariableSetType)


class QSVariableUnion(QSVariable):
    def _check_type(self):
        assert isinstance(self.type, QSVariableUnionType)

    def __getattr__(self, name):
        if self.denotations is not None:
            for i, denotation in self.type.denotations:
                if name == denotation:
                    return self.value[i]
        raise AttributeError(name)


class OverriddenCallList(list):
    """
    A data structure that holds multiple overridden __call__ implementations for a function.
    This is only useful when we are partial evaluating a function (and when the actual function type can not be
    resolved.)
    """


class ResolvedFromRecord(collections.namedtuple('_ResolvedFromRecord', ['function', 'ftype_id'])):
    pass


class QSFunction(object):
    def __init__(
        self,
        type: Union[QSFunctionType, QSOverloadedFunctionType],
        overridden_call: Optional[Union[Callable, OverriddenCallList]] = None,
        resolved_from: Optional[ResolvedFromRecord] = None,
        name=None, function_body=None
    ):

        self.type = type
        self.overridden_call = overridden_call
        self.resolved_from = resolved_from

        if isinstance(self.type, QSOverloadedFunctionType) and isinstance(self.overridden_call, OverriddenCallList):
            assert self.type.nr_types == len(self.overridden_call)

        self.name = name
        self.function_body = function_body  # the function body defined during the declaration.

    def _check_type(self):
        assert isinstance(self.type, (QSFunctionType, QSOverloadedFunctionType))

    def set_function_name(self, function_name):
        self.name = function_name

    def set_function_body(self, function_body):
        self.function_body = function_body

    @property
    def is_overloaded(self):
        return isinstance(self.type, QSOverloadedFunctionType)

    def get_overridden_call(self, ftype_id=None):
        if isinstance(self.overridden_call, OverriddenCallList):
            assert ftype_id is not None
            return self.overridden_call[ftype_id]
        return self.overridden_call

    def get_sub_function(self, ftype_id):
        assert self.is_overloaded
        assert 0 <= ftype_id < self.type.nr_types
        return type(self)(
            self.type.types[ftype_id],
            self.get_overridden_call(ftype_id),
            resolved_from=self.canonize_resolved_from(ftype_id),
            name=self.name, function_body=self.function_body[ftype_id] if self.function_body is not None else None
        )

    @cached_property
    def all_sub_functions(self):
        assert self.is_overloaded
        return [self.get_sub_function(i) for i in range(self.type.nr_types)]

    @property
    def nr_arguments(self):
        assert not self.is_overloaded
        return self.type.nr_arguments

    @classmethod
    def from_function(cls, function, implementation=True, sig=None):
        type = QSFunctionType.from_annotation(function, sig=sig)
        return cls(type, name=function.__name__, function_body=function if implementation else None)

    def __call__(self, *args, **kwargs):
        if self.overridden_call is not None:
            if isinstance(self.type, QSOverloadedFunctionType):
                ftype_id, function_type, resolved_args = self.type.resolve_type_and_args(*args, **kwargs)
                return self.get_overridden_call(ftype_id)(resolved_args)
            else:
                resolved_args = self.type.resolve_args(*args, **kwargs)
                return self.overridden_call(*resolved_args)

        if isinstance(self.type, QSOverloadedFunctionType):
            ftype_id, function_type, resolved_args = self.type.resolve_type_and_args(*args, **kwargs)
            function = QSFunction(
                function_type,
                overridden_call=None,  # Must be none.
                resolved_from=self.canonize_resolved_from(ftype_id)
            )
        else:
            resolved_args = self.type.resolve_args(*args, **kwargs)
            function = self

        return QSFunctionApplication(function, resolved_args)

    def __str__(self):
        if self.overridden_call is not None and not isinstance(self.type, QSOverloadedFunctionType):
            with QSFormatContext(type_format_cls=False).as_default():
                denotations = self.type.argument_denotations
                if denotations is None:
                    denotations = AnonymousArgumentGen().gen(self.type.nr_arguments)

                if get_format_context().function_format_lambda:
                    argument_grounding = [
                        QSFunctionObject(t, QSFunction(t, name=n)) if isinstance(t, QSFunctionType) else QSObject(t, n)
                        for t, n in zip(self.type.argument_types, denotations)
                    ]
                    fmt = ''.join(['lam ' + n + '.' for n in denotations]) + ': '
                    ret = self.overridden_call(*argument_grounding)
                    with QSFormatContext(object_format_type=False).as_default():
                        fmt += str(ret)
                else:
                    argument_grounding = [
                        QSFunctionObject(t, QSFunction(t, name=f'<{n}>')) if isinstance(t, QSFunctionType) else QSObject(t, f'<{n}>')
                        for t, n in zip(self.type.argument_types, denotations)
                    ]
                    fmt = 'def ' + self.name + '(' + ', '.join([str(x) for x in argument_grounding]) + '): '
                    ret = self.overridden_call(*argument_grounding)
                    with QSFormatContext(object_format_type=False).as_default():
                        fmt += 'return ' + indent_text(str(ret)).lstrip()
        else:
            if isinstance(self.type, QSOverloadedFunctionType):
                fmt = '\n'.join([f'{func_type}' for func_type in self.type.types])
                if self.name is not None:
                    fmt = re.sub(r'^' + re.escape(self.name) + ' (overloaded): ', '', fmt, flags=re.MULTILINE)
                    fmt = self.name + ': ' + '\n' + indent_text(fmt)
            else:
                fmt = f'{self.name}{self.type}'

        return fmt

    __repr__ = __str__

    def remap_arguments(self, remapping: List[int]):
        """
        Generate a new QSFunction object with a different argument order.
        Specifically, remapping is a permutation. The i-th argument to the new function will be the remapping[i]-th
        argument in the old function.
        """
        if isinstance(self.type, QSOverloadedFunctionType):
            raise NotImplementedError('Argument remapping for overloaded functions are not implemented.')

        new_argument_types = [self.type.argument_types[i] for i in remapping]
        new_argument_denotations = None if self.type.argument_denotations is None else [self.type.argument_denotations[i] for i in remapping]

        def new_overridden_call(*args):
            remapped_args = [None for _ in range(len(args))]
            for i, arg in enumerate(args):
                remapped_args[remapping[i]] = arg
            return self(*remapped_args)

        return QSFunction(
            QSFunctionType(
                new_argument_types, self.type.return_type,
                argument_denotations=new_argument_denotations
            ), overridden_call=new_overridden_call, resolved_from=self.resolved_from, name=self.name
        )

    def partial(self, *args, **kwargs):
        new_name = f'Partial[{self.name}]'
        new_overridden_call = None
        new_resolved_from = None

        if isinstance(self.type, QSOverloadedFunctionType):
            with QSFunctionArgumentResolutionContext(
                check_missing=False,
                check_overloaded_ambiguity=False
            ).as_default():
                types_and_arguments = self.type.resolve_type_and_args(*args, **kwargs)

            if not isinstance(types_and_arguments, QSOverloadedFunctionAmbiguousResolutions):
                ftype_id, function_type, resolved_args = types_and_arguments
                unmapped_arguments = [i for i, arg in enumerate(resolved_args) if arg is QSFunctionArgumentUnset]
                if len(unmapped_arguments) == 0:
                    return self._apply_with_resolved_args(resolved_args, ftype_id, function_type)
                new_type = self.gen_partial_function_type(function_type, unmapped_arguments)
                new_resolved_from = self.canonize_resolved_from(ftype_id)
            else:
                # Block BEGIN {{{
                # If there is one specific resolution s.t. all variables are grounded, use it.

                all_grounded_resolutions = list()
                for ftype_id, function_type, resolved_args in types_and_arguments:
                    unmapped_arguments = [i for i, arg in enumerate(resolved_args) if arg is QSFunctionArgumentUnset]
                    if len(unmapped_arguments) == 0:
                        all_grounded_resolutions.append((ftype_id, function_type, resolved_args))

                if len(all_grounded_resolutions) == 1:
                    ftype_id, function_type, resolved_args = all_grounded_resolutions[0]
                    return self._apply_with_resolved_args(resolved_args, ftype_id, function_type)
                elif len(all_grounded_resolutions) > 1:
                    with get_function_argument_resolution_context().exc():
                        fmt = 'Got ambiguous application of overloaded function{}.\n'.format(
                            '' if self.name is None else ' ' + self.name)
                        fmt += 'Candidates are:\n'
                        for r in all_grounded_resolutions:
                            fmt += indent_text(str(r[1])) + '\n'
                        fmt += 'Invoked with arguments: {}.'.format(str(all_grounded_resolutions[0][2]))
                        raise QSFunctionArgumentResolutionError(fmt)

                # }}} Block END.

                possible_resolution_ids = list()
                possible_resolutions = list()
                possible_overridden_calls = list()
                for ftype_id, function_type, resolved_args in types_and_arguments:
                    unmapped_arguments = [i for i, arg in enumerate(resolved_args) if arg is QSFunctionArgumentUnset]
                    new_subtype = self.gen_partial_function_type(
                        function_type, unmapped_arguments
                    )
                    possible_resolution_ids.append(ftype_id)
                    possible_resolutions.append(new_subtype)
                    possible_overridden_calls.append(self.gen_partial_overriden_call(
                        function_type, new_subtype, resolved_args, self
                    ))
                new_type = QSOverloadedFunctionType(possible_resolutions)
                new_resolved_from = self.canonize_resolved_from(possible_resolution_ids)
                new_overridden_call = OverriddenCallList(possible_overridden_calls)
        else:
            with QSFunctionArgumentResolutionContext(check_missing=False).as_default():
                resolved_args = self.type.resolve_args(*args, **kwargs)
            unmapped_arguments = [i for i, arg in enumerate(resolved_args) if arg is QSFunctionArgumentUnset]

            if len(unmapped_arguments) == 0:
                return self._apply_with_resolved_args(resolved_args)

            new_type = self.gen_partial_function_type(self.type, unmapped_arguments)
            new_overridden_call = self.gen_partial_overriden_call(
                self.type, new_type, resolved_args, self
            )

        return QSFunction(new_type, overridden_call=new_overridden_call, resolved_from=new_resolved_from, name=new_name)

    def canonize_resolved_from(self, ftype_id):
        # if self.resolved_from is not None:
        #     assert self.resolved_from[1] is None
        #     return self.resolved_from[0].canonize_resolved_from(ftype_id)
        return ResolvedFromRecord(self, ftype_id)

    def _apply_with_resolved_args(
            self, resolved_args,
            resolved_ftype_id=None, resolved_function_type=None
    ):

        if self.overridden_call is not None:
            return self.get_overridden_call(resolved_ftype_id)(*resolved_args)

        if isinstance(self.type, QSOverloadedFunctionType):
            function = QSFunction(
                resolved_function_type,
                overridden_call=None,  # Must be none.
                resolved_from=self.canonize_resolved_from(resolved_ftype_id)
            )
        else:
            function = self
        return QSFunctionApplication(function, resolved_args)

    @staticmethod
    def gen_partial_function_type(old_type, unmapped_arguments):
        new_argument_types = [old_type.argument_types[i] for i in unmapped_arguments]
        new_return_type = old_type.return_type
        new_argument_denotations = None
        if old_type.argument_denotations is not None:
            new_argument_denotations = [old_type.argument_denotations[i] for i in unmapped_arguments]
        new_type = QSFunctionType(new_argument_types, new_return_type, argument_denotations=new_argument_denotations)
        return new_type

    @staticmethod
    def gen_partial_overriden_call(old_type, new_type, resolved_args, call):
        assert isinstance(new_type, QSFunctionType)

        def partial_overriden_call(*new_args, **new_kwargs):
            new_resolved_args = new_type.resolve_args(*new_args, **new_kwargs)
            new_full_args = resolved_args.copy()
            j = 0
            for i in range(len(resolved_args)):
                if new_full_args[i] is QSFunctionArgumentUnset:
                    new_full_args[i] = new_resolved_args[j]
                    j += 1
            return call(*new_full_args)

        return partial_overriden_call


class QSFunctionApplication(object):
    def __init__(self, function: QSFunction, args):
        self.function = function
        self.args = args

    @property
    def return_type(self):
        return self.function.type.return_type

    def __str__(self):
        fmt = self.function.name + '('
        arg_fmt = [str(x) for x in self.args]
        arg_fmt_len = [len(x) for x in arg_fmt]

        ctx = get_format_context()

        # The following criterion is just an approximation. A more principled way is to pass the current indent level
        # to the recursive calls to str(x).
        if ctx.expr_max_length > 0 and (sum(arg_fmt_len) + len(fmt) + 1 > ctx.expr_max_length):
            if sum(arg_fmt_len) > ctx.expr_max_length:
                fmt += '\n' + ',\n'.join([indent_text(x) for x in arg_fmt]) + '\n'
            else:
                fmt += '\n' + ', '.join(arg_fmt) + '\n'
        else:
            fmt += ', '.join(arg_fmt)
        fmt += ')'
        return fmt

    __repr__ = __str__


class AnonymousArgument(object):
    def __init__(self, name=None):
        self.name = name

    def __str__(self):
        return 'Anonymous[' + str(self.name) + ']'

    __repr__ = __str__


class AnonymousArgumentGen(object):
    def __init__(self, format='z{i:d}'):
        self.format = format
        self.counter = 0

    @property
    def nr_generated(self):
        return self.counter

    def gen(self, n=None):
        if n is None:
            self.counter += 1
            return self.format.format(i=self.counter)
        return [self.gen() for _ in range(n)]


class _QSTypingFunctionSugarInner(object):
    def __init__(self, return_type):
        self.return_type = return_type

    def __call__(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return QSFunctionType(tuple(), self.return_type)
        elif len(args) != 0:
            assert len(kwargs) == 0, 'Only support all positional arguments or all positional keyword arguments.'
            return QSFunctionType(args, self.return_type)
        elif len(kwargs) != 0:
            assert len(args) == 0, 'Only support all positional arguments or all positional keyword arguments.'
            return QSFunctionType(tuple(kwargs.values()), self.return_type, tuple(kwargs.keys()))
        raise ValueError('Unreachable.')


class _QSTypingFunctionSugar(object):
    def __getitem__(self, return_type):
        return _QSTypingFunctionSugarInner(return_type)


QSTypingFunction = _QSTypingFunctionSugar()


class QSTypeSystem(object):
    def __init__(self, name='<QSTypeSystem>'):
        self.name = name
        self.types = dict()
        self.constants = dict()
        self.functions = dict()

        self.__define_overloaded_functions = collections.defaultdict(list)

    @property
    def ft_(self):
        return QSTypingFunction

    def define_type(self, type):
        if type.typename is None:
            raise ValueError('Cannot define anonymous types (such as overloaded function types or typing unions.')
        self.types[type.typename] = type

    def define_const(self, type, value):
        if isinstance(type, QSConstantType):
            self.constants[value] = QSConstant(type, value)
        elif isinstance(type, QSVariableType):
            self.constants[value] = QSVariable(type, value)
        else:
            raise TypeError('Invalid constant type: {}.'.format(type))

    def define_function(self, function, implementation=True):
        if not isinstance(function, QSFunction):
            assert inspect.isfunction(function)
            function = QSFunction.from_function(function, implementation=implementation)
        self.functions[function.name] = function

    def define_overloaded_function(self, name, overloads, implementation=True):
        overloads_qs = list()
        overloads_function_body = list()
        for f in overloads:
            if isinstance(f, QSFunction):
                overloads_qs.append(f.type)
                overloads_function_body.append(f.function_body)
            else:
                overloads_qs.append(QSFunctionType.from_annotation(f))
                overloads_function_body.append(f)
        self.functions[name] = QSFunction(
            QSOverloadedFunctionType(overloads_qs),
            overridden_call=None, resolved_from=None,
            name=name, function_body=overloads_function_body if implementation else None
        )

    def lam(self, lambda_expression, name='<lambda>', typing_cues=None):
        sig = inspect.signature(lambda_expression)
        argument_denotations = list(sig.parameters.keys())

        if len(argument_denotations) == 0:
            return lambda_expression()

        try:
            function_type = QSFunctionType.from_annotation(lambda_expression)
        except QSFunctionArgumentResolutionError:
            function_type = self._resolve_lambda_function_type(lambda_expression, typing_cues)

        return QSFunction(function_type, overridden_call=lambda_expression, name=name)

    def canonize_type(self, type):
        if type is None:
            return type
        if isinstance(type, QSType):
            return type
        if isinstance(type, QSTypingUnionType):
            return QSTypingUnionType(tuple(self.canonize_type(t) for t in type.types))
        assert isinstance(type, six.string_types)
        return self.types[type]

    def canonize_signature(self, signature: inspect.Signature):
        params = [
            inspect.Parameter(v.name, v.kind, default=v.default, annotation=self.canonize_type(v.annotation))
            for k, v in signature.parameters.items()
        ]
        return_annotation = self.canonize_type(signature.return_annotation)
        return inspect.Signature(params, return_annotation=return_annotation)

    def _resolve_lambda_function_type(self, lambda_expression, typing_cues):
        sig = inspect.signature(lambda_expression)
        parameters = list(sig.parameters.keys())
        parameter_types = {k: AnyType for k in parameters}

        for i, (name, param) in enumerate(sig.parameters.items()):
            if param.annotation is not sig.empty:
                parameter_types[param.name] = param.annotation
        return_type = sig.return_annotation if sig.return_annotation is not sig.empty else AnyType

        if typing_cues is not None:
            for k, v in typing_cues.items():
                assert k in parameter_types
                parameter_types[k] = v

            if 'return' in typing_cues:
                return_type = typing_cues['return']

        for k in sig.parameters:
            if parameter_types[k] is AnyType:
                exceptions = list()
                success_types = list()

                for k_type in self.types.values():
                    parameter_types[k] = k_type
                    parameter_grounding = list()

                    for param in parameters:
                        t = parameter_types[param]
                        parameter_grounding.append(QSFunctionObject(t, QSFunction(t)) if isinstance(t, QSFunctionType) else QSObject(t, None))

                    try:
                        output = lambda_expression(*parameter_grounding)
                        success_types.append(k_type)

                        if return_type is AnyType:
                            return_type = get_type(output)

                    except QSFunctionArgumentResolutionError as e:
                        exceptions.append((k_type, e))

                if len(success_types) == 1:
                    parameter_types[k] = success_types[0]
                elif len(success_types) == 0:
                    with get_function_argument_resolution_context().exc():
                        fmt = 'Failed to infer argument type for {}.\n'.format(k)
                        fmt += 'Detailed messages are:\n'
                        for t, e in exceptions:
                            fmt += indent_text('Trying {}:\n'.format(t) + indent_text(str(e)) + '\n')
                        raise QSFunctionArgumentResolutionError(fmt.rstrip())
                else:
                    with get_function_argument_resolution_context().exc():
                        fmt = 'Got ambiguous type for {}.\n'.format(k)
                        fmt += 'Candidates are:\n'
                        for r in success_types:
                            fmt += indent_text(str(r)) + '\n'
                        raise QSFunctionArgumentResolutionError(fmt.strip())

        return QSFunctionType([parameter_types[i] for i in parameters], return_type, parameters)

    def __getattr__(self, name):
        if name.startswith('t_'):
            return self.types[name[2:]]
        elif name.startswith('c_'):
            return self.constants[name[2:]]
        elif name.startswith('f_'):
            return self.functions[name[2:]]
        raise AttributeError(name)

    @contextlib.contextmanager
    def define(self, implementation=True):
        locals_before = inspect.stack()[2][0].f_locals.copy()
        annotations_before = locals_before.get('__annotations__', dict()).copy()
        yield self
        locals_after = inspect.stack()[2][0].f_locals.copy()
        annotations_after = locals_after.get('__annotations__', dict()).copy()

        new_functions = {
            k for k in locals_after
            if k not in locals_before and not isinstance(locals_after[k], QSTypeSystem)
        }
        new_annotations = {
            k for k in annotations_after
            if k not in annotations_before or annotations_after[k] != annotations_before[k]
        }

        if len(new_annotations) == 0:
            logger.warning('ts.define() for constants is only allowed at the global scope.')

        functions = list()
        for func_name in new_functions:
            var = locals_after[func_name]
            if not inspect.isfunction(var):
                raise ValueError('Support only function definitions in the DEFINE body, got {}.'.format(func_name))
            if func_name in self.__define_overloaded_functions:
                continue
            functions.append((var.__code__.co_firstlineno, var))

        for func_name, overloads in self.__define_overloaded_functions.items():
            lineno = overloads[0].__code__.co_firstlineno
            if len(overloads) > 1:
                functions.append((lineno, (func_name, overloads)))
            else:
                functions.append((lineno, overloads[0]))
        functions.sort()

        for _, f in functions:
            if isinstance(f, tuple):
                self.define_overloaded_function(*f, implementation=implementation)
            else:
                self.define_function(f, implementation=implementation)

        for const_name in new_annotations:
            variable_type = annotations_after[const_name]
            self.define_const(variable_type, const_name)

        self.__define_overloaded_functions = collections.defaultdict(list)

    def overload(self, function):
        self.__define_overloaded_functions[function.__name__].append(function)
        return function

    def __str__(self):
        fmt = 'Types:\n'
        for type in self.types.values():
            fmt += '  ' + str(type) + '\n'
        fmt += 'Constants:\n'
        for const in self.constants.values():
            fmt += '  ' + str(const) + '\n'
        fmt += 'Functions:\n'
        for function in self.functions.values():
            fmt += '  ' + str(function).replace('\n', '\n  ') + '\n'

        fmt = 'TypeSystem: {}\n'.format(self.name) + indent_text(fmt.rstrip())
        return fmt

    __repr__ = __str__

    def serialize(self, program):
        def dfs(p):
            if isinstance(p, QSFunctionApplication):
                record = dict()
                record['__serialize_type__'] = 'function'
                record['function'] = p.function.name
                record['args'] = list()
                for arg in p.args:
                    record['args'].append(dfs(arg))
            elif isinstance(p, QSObject):
                record = dict()
                record['__serialize_type__'] = 'object'
                record['type'] = p.type.typename
                record['value'] = p.value
            else:
                raise TypeError('Unserializable object: {}.'.format(type(p)))
            return record
        return dfs(program)

    def deserialize(self, dictionary):
        def dfs(d):
            assert '__serialize_type__' in d
            stype = d['__serialize_type__']
            if stype == 'function':
                func = self.functions[d['function']]
                args = list()
                for arg in d['args']:
                    args.append(dfs(arg))
                return func(*args)
            elif stype == 'object':
                return QSObject(self.types[d['type']], d['value'])
            else:
                raise TypeError('Unrecogniable serialized type: {}.'.format(stype))
        return dfs(dictionary)
