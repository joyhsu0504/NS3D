#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Joy Hsu <joycj@stanford.edu>
#
# Distributed under terms of the MIT license.

import torch.nn as nn
import jactorch.nn as jacnn

from models.modules.referit3d_concept import ReferIt3DConceptEmbeddingV1
from jacinle.config.environ_v2 import configs, def_configs_func
from jacinle.logging import get_logger
from datasets.definition import gdef

logger = get_logger(__file__)

__all__ = ['NS3DReferIt3DModelBase', 'NS3DReferIt3DModel']


class NS3DReferIt3DModelBase(nn.Module):
    @staticmethod
    @def_configs_func
    def _def_configs():
        # model configs for scene graph
        configs.model.sg_dims = [None, 256, 256, 256]

        # model configs for visual-semantic embeddings
        pn_output_dim = 128
        output_dim = 607
        configs.model.vse_hidden_dims = [None, output_dim, pn_output_dim, pn_output_dim*3]

        # supervision configs
        configs.train.weight_decay = 0
        configs.train.scene_add_supervision = False
        configs.train.refexp_add_supervision = True

        return configs

    def __init__(self):
        super().__init__()
        self._def_configs()

        import models.scene_graph.scene_graph_pointnet as sng
        pn_output_dim = 128
        output_dim = 607
        self.scene_graph = sng.SceneGraphPointNet(pn_output_dim, output_dim)

        self.attribute_embedding = ReferIt3DConceptEmbeddingV1()
        self.relation_embedding = ReferIt3DConceptEmbeddingV1()
        self.multi_relation_embedding = ReferIt3DConceptEmbeddingV1()
        
        self.init_concept_embeddings()

    def train(self, mode=True):
        super().train(mode)

    def init_concept_embeddings(self):       
        for arity, src, tgt in zip(
            [1, 2, 3],
            [gdef.attribute_concepts, gdef.relational_concepts, gdef.multi_relational_concepts],
            [self.attribute_embedding, self.relation_embedding, self.multi_relation_embedding]
        ):
            tgt.init_attribute('all', configs.model.sg_dims[arity])
            for word in src:
                tgt.init_concept(word, configs.model.vse_hidden_dims[arity], 'all')

    def forward_sng(self, feed_dict):
        f_sng = self.scene_graph(feed_dict.scene, feed_dict.input_objects, feed_dict.input_objects_length)
        f_sng = [
            {'attribute': sng[1], 'relation': sng[2], 'multi_relation': sng[3]}
            for sng in f_sng
        ]
        return f_sng


class NS3DReferIt3DModel(NS3DReferIt3DModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from models.modules.referit3d_reasoning import ReferIt3DQSGrounding, ReferIt3DQSExecutor
        self.grounding_cls = ReferIt3DQSGrounding
        self.executor = ReferIt3DQSExecutor(gdef.type_system)

        from models.modules.referit3d_reasoning import SceneConceptLoss, RefExpLoss
        self.scene_concept_loss = SceneConceptLoss(add_supervision=configs.train.scene_add_supervision)
        self.refexp_loss = RefExpLoss(add_supervision=configs.train.refexp_add_supervision)