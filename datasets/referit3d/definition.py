#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Joy Hsu <joycj@stanford.edu>
#
# Distributed under terms of the MIT license.

import six

from enum import IntEnum
from jacinle.utils.cache import cached_property
from jacinle.utils.enum import JacEnum

from datasets.definition import DatasetDefinitionBase
import semantics.typing as T
from semantics.ccg import CCG
from semantics.syntax import CCGSyntaxSystem

__all__ = ['ReferIt3DDefinition']


class ReferIt3DDatasetSplit(JacEnum):
    ORIG = 'orig'


class ReferIt3DDefinition(DatasetDefinitionBase):
    def __init__(self):
        import pickle
        referit3dnet_class_to_idx = pickle.load(open('datasets/referit3d/data/referit3dnet_class_to_idx.p', 'rb'))
        nouns = list(referit3dnet_class_to_idx.keys())

        parsed_nouns = []
        for n in nouns:
            n = n.replace(' ', '_')
            parsed_nouns.append(n)

        self.nouns = parsed_nouns
        self.attribute_concepts = self.nouns + ['clutter', 'error_it', 'error_them']
        self.multi_relational_concepts = ['between', 'center', 'middle', 'facing', 'looking', 'left', 'right', 'behind', 'back']
        self.relational_concepts = ['on','left', 'right', 'front', 'behind', 'above', 'below',
                'beside', 'over', 'under', 'beneath', 'underneath', 'lying', 'next',
                'back', 'top', 'supporting','with',
                'near', 'close', 'closer', 'closest', 'far', 'farthest']

    @cached_property
    def type_system(self):
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

        return ts

    def canonize_answer(self, output, device=None):
        return T.QSObject(self.type_system.t_action, output)

