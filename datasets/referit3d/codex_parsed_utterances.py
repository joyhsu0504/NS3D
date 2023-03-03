#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Joy Hsu <joycj@stanford.edu>
#
# Distributed under terms of the MIT license.

import lark
import semantics.typing as T

ts = T.QSTypeSystem('ReferIt3D')
ts.define_type(T.QSVariableType('object_set'))
ts.define_type(T.QSConstantType('object_property'))
ts.define_type(T.QSConstantType('object_relation'))

with ts.define(implementation=False):
    def scene() -> ts.t_object_set: ...
    def filter(obj: ts.t_object_set, p: ts.t_object_property) -> ts.t_object_set: ...
    def relate(obj_left: ts.t_object_set, obj_right: ts.t_object_set, r: ts.t_object_relation) -> ts.t_object_set: ...
    def relate_multi(obj_target: ts.t_object_set, obj_side1: ts.t_object_set, obj_side2: ts.t_object_set, r: ts.t_object_relation) -> ts.t_object_set: ...
    def relate_anchor(obj_target: ts.t_object_set, obj_side: ts.t_object_set, obj_anchor: ts.t_object_set, r: ts.t_object_relation) -> ts.t_object_set: ...
    def anchor(anchor: ts.t_object_set, main: ts.t_object_set) -> ts.t_object_set: ...


grammar = r"""
start: function_application
function_application: function_name "(" (argument ("," argument)*)? ")"
function_name: STRING
argument: function_application | constant

constant: STRING

%import common.LETTER
%import common.DIGIT
STRING: LETTER ("_"|"-"|LETTER|DIGIT)*

%import common.WS
%ignore WS
"""

inline_args = lark.v_args(inline=True)


class FunctionalTransformer(lark.Transformer):
    def start(self, args):
        return args[0]

    @inline_args
    def function_application(self, function_name, *args):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, str):
                if function_name == 'filter':
                    args[i] = T.QSConstant(ts.types['object_property'], arg)
                else:
                    args[i] = T.QSConstant(ts.types['object_relation'], arg)

        return ts.functions[function_name](*args)

    def function_name(self, function_name):
        return function_name[0].value

    def argument(self, argument):
        return argument[0]

    def constant(self, constant):
        return constant[0].value



def parse_codex_text(parser, trans, text):
    tree = parser.parse(text)
    return trans.transform(tree)

