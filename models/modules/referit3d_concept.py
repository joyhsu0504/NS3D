#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Joy Hsu <joycj@stanford.edu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import jacinle
import jactorch

from models.utils.visual_reasoning_utils import VisualConceptEmbeddingBase

__all__ = ['ReferIt3DConceptEmbeddingV1']


class _AttributeCrossBlock(nn.Module):
    def __init__(self, name, embedding_dim):
        super().__init__()

        self.name = name
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(embedding_dim * 4, 1)

    def forward(self, a, b=None):
        if b is None:
            a, b = jactorch.meshgrid(a, dim=-2)

        c = torch.cat((a, b, a*b, a-b), dim=-1)
        return self.embedding(c).squeeze(-1)


class _ConceptBlock(nn.Module):
    """
    Concept as an embedding in the corresponding attribute space.
    """
    def __init__(self, name, embedding_dim, nr_attributes):
        """

        Args:
            name (str): name of the concept.
            embedding_dim (int): dimension of the embedding.
            nr_attributes (int): number of known attributes.
        """
        super().__init__()

        self.name = name
        self.embedding_dim = embedding_dim
        self.nr_attributes = nr_attributes
        self.embedding = nn.Parameter(torch.randn(embedding_dim))

        self.belong = nn.Parameter(torch.randn(nr_attributes) * 0.1)
        self.known_belong = False

    def set_belong(self, belong_id):
        """
        Set the attribute that this concept belongs to.

        Args:
            belong_id (int): the id of the attribute.
        """
        self.belong.data.fill_(-100)
        self.belong.data[belong_id] = 100
        self.belong.requires_grad = False
        self.known_belong = True

    @property
    def log_normalized_belong(self):
        """Log-softmax-normalized belong vector."""
        return F.log_softmax(self.belong, dim=-1)

    @property
    def normalized_belong(self):
        """Softmax-normalized belong vector."""
        return F.softmax(self.belong, dim=-1)


class ReferIt3DConceptEmbeddingV1(VisualConceptEmbeddingBase):
    def __init__(self, enable_cross_similariy=False):
        super().__init__()

        self.enable_cross_similariy = enable_cross_similariy

        self.all_attributes = list()
        self.all_concepts = list()
        self.attribute_cross_embeddings = nn.Module()
        self.concept_embeddings = nn.Module()

    @property
    def nr_attributes(self):
        return len(self.all_attributes)

    @property
    def nr_concepts(self):
        return len(self.all_concepts)

    @jacinle.cached_property
    def attribute2id(self):
        return {a: i for i, a in enumerate(self.all_attributes)}

    def init_attribute(self, identifier, input_dim=None):
        assert self.nr_concepts == 0, 'Can not register attributes after having registered any concepts.'
        self.all_attributes.append(identifier)
        self.all_attributes.sort()

        if self.enable_cross_similariy:
            assert input_dim is not None
            block = _AttributeCrossBlock(identifier, input_dim)
            self.attribute_cross_embeddings.add_module('attribute_' + identifier, block)

    def init_concept(self, identifier, input_dim, known_belong=None):
        block = _ConceptBlock(identifier, input_dim, self.nr_attributes)
        self.concept_embeddings.add_module('concept_' + identifier, block)
        self.all_concepts.append(identifier)
        if known_belong is not None:
            block.set_belong(self.attribute2id[known_belong])

    def get_belongs(self):
        """
        Return a dict which maps from all attributes (by name) to a list of concepts (by name).
        """
        belongs = dict()
        for k, v in self.concept_embeddings.named_children():
            belongs[k] = self.all_attributes[v.belong.argmax(-1).item()]
        class_based = dict()
        for k, v in belongs.items():
            class_based.setdefault(v, list()).append(k)
        class_based = {k: sorted(v) for k, v in class_based.items()}
        return class_based

    def get_attribute_cross(self, identifier):
        return getattr(self.attribute_cross_embeddings, 'attribute_' + identifier)

    def get_concept(self, identifier):
        return getattr(self.concept_embeddings, 'concept_' + identifier)

    def get_all_concepts(self):
        return {c: self.get_concept(c) for c in self.all_concepts}

    def get_concepts_by_attribute(self, identifier):
        return self.get_all_concepts(), self.attribute2id[identifier]

    def similarity(self, query, identifier):
        concept = self.get_concept(identifier)
        reference = jactorch.add_dim_as_except(concept.embedding, query, -1)
        logits = (query * reference).sum(dim=-1)
        return logits

    def cross_similarity(self, query, identifier):
        mapping = self.get_attribute_cross(identifier)
        logits = mapping(query)
        return logits

    def query_attribute(self, query, identifier):
        concepts, attr_id = self.get_concepts_by_attribute(identifier)

        word2idx = {}
        masks = []
        for k, v in concepts.items():
            embedding = jactorch.add_dim_as_except(v.embedding, query, -1)
            mask = (query * embedding).sum(dim=-1)

            belong_score = v.log_normalized_belong[attr_id]
            mask = mask + belong_score

            masks.append(mask)
            word2idx[k] = len(word2idx)

        masks = torch.stack(masks, dim=-1)
        return masks, word2idx

