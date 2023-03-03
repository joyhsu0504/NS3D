#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : implementation.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import inspect
from typing import Union
from semantics.typing import QSFunction, QSObject, QSFunctionApplication
import jacinle

__all__ = ['QSImplementation', 'QSImplementationRegistrationError', 'qs_overload_impl']


class QSImplementationRegistrationError(TypeError):
    pass


class QSImplementationExecutionError(Exception):
    pass


class QSImplementation(object):
    def __init__(self, type_system):
        self.type_system = type_system
        self.registered_functions = dict()
        self._register_functions()

    def __call__(self, value: Union[QSFunction, QSFunctionApplication], *args, **kwargs):
        if isinstance(value, QSFunction):
            value = value(*args, **kwargs)
        elif callable(value):
            value = value(*args, **kwargs)

        def dfs(v):
            if isinstance(v, QSObject):
                return v
            assert isinstance(v, QSFunctionApplication)
            args = [dfs(x) for x in v.args]
            summary = self._get_function_application_summary(v)
            if summary not in self.registered_functions:
                raise NotImplementedError('Not implemented function: {}.'.format(v.function.type))
            ftype, function_call = self.registered_functions[summary]
            ret = function_call(*args)
            if isinstance(ret, QSObject):
                return ret
            return QSObject(ftype.return_type, ret)

        try:
            return dfs(value)
        finally:
            del dfs

    def compile(self, function: QSFunction):
        def compiled_func(*args, **kwargs):
            return self(function, *args, **kwargs)
        return compiled_func

    def _register_functions(self):
        for fname in dir(self):
            f = getattr(self, fname)
            if fname in self.type_system.functions:
                tsf = self.type_system.functions[fname]
                if isinstance(f, QSImplementationOverloadedSubFunction):
                    pass
                elif isinstance(f, QSImplementationOverloadedFunction):
                    if not tsf.is_overloaded:
                        raise QSImplementationRegistrationError(
                            f'Function {fname} is not an overloaded function in the type system.'
                        )
                    for subf in f.overloaded_impls:
                        self._register_function(subf, tsf)
                else:
                    self._register_function(f, tsf)

    def _register_function(self, f, tsf):
        """

        Args:
            f (callable): the actual implementation.
            tsf: the function object defined in the type system.

        Returns: None

        """
        def run_match(function_name, ftype, tsf):
            matched_ftypes = list()
            if tsf.is_overloaded:
                for i, subtype2 in enumerate(tsf.type.types):
                    if ftype.eq_arguments(subtype2):
                        matched_ftypes.append(i)
            else:
                if ftype.eq_arguments(tsf.type):
                    matched_ftypes.append(0)
            if len(matched_ftypes) == 1:
                add_matched_function(function_name, ftype, f)
            elif len(matched_ftypes) == 0:
                logger = jacinle.get_logger(__file__)
                logger.warning('Unknown signature for function: {}. Going to ignore this implementation.'.format(ftype))
            else:
                raise QSImplementationRegistrationError('Ambiguous signature for function: {}.'.format(ftype))

        def add_matched_function(function_name, ftype, function):
            summary = self._get_function_summary(function_name, ftype)
            if summary in self.registered_functions:
                raise QSImplementationRegistrationError('Duplicated registration for function: {}.'.format(ftype))
            self.registered_functions[summary] = (ftype, function)

        sig = self.type_system.canonize_signature(inspect.signature(f))
        qsf = QSFunction.from_function(f, sig=sig)
        if qsf.is_overloaded:
            for subtype in qsf.type.types:
                run_match(qsf.name, subtype, tsf)
        else:
            run_match(qsf.name, qsf.type, tsf)

    @staticmethod
    def _get_function_summary(function_name, ftype):
        summary = [function_name]
        for arg in ftype.argument_types:
            summary.append(str(arg))
        return tuple(summary)

    @staticmethod
    def _get_function_application_summary(fa):
        assert not fa.function.is_overloaded
        return QSImplementation._get_function_summary(fa.function.name, fa.function.type)


class QSImplementationFromTypeDef(QSImplementation):
    def _register_functions(self):
        for function_name, function in self.type_system.functions.items():
            if function.is_overloaded:
                for subfunc in function.all_sub_functions:
                    summary = self._get_function_summary(subfunc.name, subfunc.type)
                    self.registered_functions[summary] = (subfunc.type, subfunc.function_body)
            else:
                summary = self._get_function_summary(function.name, function.type)
                self.registered_functions[summary] = (function.type, function.function_body)


class QSImplementationOverloadedSubFunction(object):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __call__(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)


class QSImplementationOverloadedFunction(object):
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.overloaded_impls = list()

    def overload(self, f):
        self.overloaded_impls.append(f)
        return QSImplementationOverloadedSubFunction(f)

    def __call__(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)


def qs_overload_impl(wrapped):
    return QSImplementationOverloadedFunction(wrapped)