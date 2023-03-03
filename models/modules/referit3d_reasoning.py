#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Joy Hsu <joycj@stanford.edu>
#
# Distributed under terms of the MIT license.

import contextlib
import collections

import torch
import torch.nn.functional as F
import jactorch
import jactorch.nn as jacnn
from semantics.implementation import QSImplementation
from models.utils.visual_reasoning_utils import VisualConceptInferenceCache
from models.losses import MultitaskLossBase, MultilabelSigmoidCrossEntropyAndAccuracy
from datasets.definition import gdef

__all__ = ['ReferIt3DQSGrounding', 'ReferIt3DQSExecutor', 'RefExpLoss', 'SceneConceptLoss']


class ReferIt3DQSGrounding(VisualConceptInferenceCache):
    def __init__(self, raw_features, embedding_registry, training, input_objects_class, class_to_idx):
        super().__init__()
        self.raw_features = raw_features
        self.embedding_registry = embedding_registry
        self.train(training)
        
        self.input_objects_class = input_objects_class
        cleaned_class_to_idx = {}
        for k in class_to_idx.keys():
            cleaned_class_to_idx[k.replace(' ', '_')] = class_to_idx[k]
        self.class_to_idx = cleaned_class_to_idx        

    def ones_value(self, *shape, log=True):
        offset = 10 if log else 1
        tensor = torch.zeros(shape, dtype=torch.float32, device=self.get_device()) + offset
        return tensor

    def get_device(self):
        return self.raw_features['attribute'].device

    def get_nr_objects(self):
        return self.raw_features['attribute'].size(0)
    
    @VisualConceptInferenceCache.cached_result('filter')
    def infer_filter(self, concept_cat, concept):
        if concept_cat == 'attribute':
            idx = self.class_to_idx[concept]
            mask = self.raw_features[concept_cat][:, idx]
        elif concept_cat == 'multi_relation':
            feat = self.raw_features['multi_relation']
            object_len = feat.size(0)
            feat = torch.cat([
                jactorch.add_dim(feat, 0, object_len),
                jactorch.add_dim(feat, 1, object_len),
                jactorch.add_dim(feat, 2, object_len)
            ], dim=3)
            mask = self.get_embedding_mod(concept_cat).similarity(feat, concept)
        else: # relation
            mask = self.get_embedding_mod(concept_cat).similarity(self.raw_features[concept_cat], concept)
            
        return mask

    @VisualConceptInferenceCache.cached_result('same')
    def infer_same(self, concept_cat, attribute):
        mask = self.get_embedding_mod(concept_cat).cross_similarity(self.raw_features[concept_cat], attribute)
        return mask

    def get_embedding_mod(self, concept_cat):
        embedding_mod_name = concept_cat + '_embedding'
        return getattr(self.embedding_registry, embedding_mod_name)


class ReferIt3DQSExecutor(QSImplementation):
    def __init__(self, type_system):
        super().__init__(type_system)
        self.grounding = None

    @contextlib.contextmanager
    def with_grounding(self, grounding):
        self.grounding = grounding
        yield
        self.grounding = None

    def scene(self) -> 'object_set':
        return self.grounding.ones_value(self.grounding.get_nr_objects())

    def filter(self, obj: 'object_set', p: 'object_property') -> 'object_set':
        mask = self.grounding.infer_filter('attribute', p.value)
        return torch.min(obj.value, mask)
    
    def relate(self, obj_left: 'object_set', obj_right: 'object_set', r: 'object_relation') -> 'object_set':
        mask = F.softmax(obj_right.value, dim=-1) @ _do_apply_self_mask(self.grounding.infer_filter('relation', r.value))
        return torch.min(obj_left.value, mask)
    
    def relate_multi(self, obj_target: 'object_set', obj_side1: 'object_set', obj_side2: 'object_set', r: 'object_relation') -> 'object_set':
        between_mask = self.grounding.infer_filter('multi_relation', r.value)
        mask = torch.einsum('ijk,j,k->i', between_mask, F.softmax(obj_side1.value), F.softmax(obj_side2.value))
        return torch.min(obj_target.value, mask)

    def relate_anchor(self, obj_target: 'object_set', obj_side: 'object_set', obj_anchor: 'object_set', r: 'object_relation') -> 'object_set':
        anchor_mask = self.grounding.infer_filter('multi_relation', r.value) 
        mask = torch.einsum('ijk,j,k->i', anchor_mask, F.softmax(obj_side.value), F.softmax(obj_anchor.value))
        return torch.min(obj_target.value, mask)


def _get_self_mask(m):
    self_mask = torch.eye(m.size(-1), dtype=m.dtype, device=m.device)
    return self_mask


def _do_apply_self_mask(m):
    self_mask = _get_self_mask(m)
    return m * (1 - self_mask) + (-10) * self_mask


def _logit_softmax(tensor, dim=-1):
    return F.log_softmax(tensor, dim=dim) - torch.log1p(-F.softmax(tensor, dim=dim).clamp(max=1-1e8))


class RefExpLoss(jacnn.TorchApplyRecorderMixin, MultitaskLossBase):
    def __init__(self, add_supervision=True, softmax=False, one_hot=False):
        super().__init__()
        self.add_supervision = add_supervision
        self.softmax = softmax
        self.one_hot = one_hot
        self._multilabel_sigmoid_xent_loss_acc = MultilabelSigmoidCrossEntropyAndAccuracy(
            compute_loss=self.add_supervision,
            softmax=softmax,
            one_hot=self.one_hot
        )

    def forward(self, input, target, input_objects_length):
        monitors = dict()
        outputs = dict()

        batch_size = len(input)
        loss, acc, acc_instance = 0, 0, 0
        for i in range(batch_size):
            this_input = input[i]
            this_target = target[i]

            l = self._batched_xent_loss(this_input, this_target)
            a = float(torch.argmax(this_input) == this_target)     
            ai = (this_input[this_target] > 0).float()

            loss += l
            acc += a
            acc_instance += ai

        if self.training and self.add_supervision:
            monitors['loss/refexp'] = loss / batch_size
            
        if self.training:
            monitors['acc/refexp'] = acc / batch_size
            monitors['acc/refexp/instance'] = acc_instance / batch_size
        else:
            monitors['validation/acc/refexp'] = acc / batch_size
            monitors['validation/acc/refexp/instance'] = acc_instance / batch_size
        
        return monitors, outputs


class SceneConceptLoss(MultitaskLossBase):
    def __init__(self, add_supervision=False):
        super().__init__()
        self.add_supervision = add_supervision

    def forward(self, feed_dict, f_sng, attribute_embedding, referred_objs, reference_objs):
        outputs, monitors = dict(), dict()

        objects = [f['attribute'] for f in f_sng]
        all_f = torch.stack(objects)
        object_labels = feed_dict['input_objects_class']
        idx_to_class = {v: k for k, v in feed_dict['class_to_idx'][0].items()}
        
        if not self.training: # logs
            objs_pred_as_referred_objs, objs_pred_as_reference_objs = [], []
            for b in range(all_f.size(0)):
                this_object_pred = all_f[b, :, :]
                this_referred_obj = feed_dict['class_to_idx'][0][referred_objs[b].replace('_', ' ')]
                this_reference_obj = feed_dict['class_to_idx'][0][reference_objs[b].replace('_', ' ')]

                objs_pred_as_referred_obj_orig = this_object_pred[:, this_referred_obj]
                objs_pred_as_reference_obj_orig = this_object_pred[:, this_reference_obj]
                objs_pred_as_referred_obj_orig = F.softmax(objs_pred_as_referred_obj_orig)
                objs_pred_as_reference_obj_orig = F.softmax(objs_pred_as_reference_obj_orig)
                objs_pred_as_referred_obj = list(range(objs_pred_as_referred_obj_orig.size(0)))
                objs_pred_as_reference_obj = list(range(objs_pred_as_reference_obj_orig.size(0)))
                this_obj_labels = object_labels[b]
                this_obj_labels = [idx_to_class[int(o_l.cpu().numpy())] for o_l in this_obj_labels]

                objs_pred_as_referred_obj_vals = [str(round(float(objs_pred_as_referred_obj_orig[o_i].cpu().numpy()), 4)) for o_i in objs_pred_as_referred_obj]
                objs_pred_as_referred_obj = [this_obj_labels[o_i] for o_i in objs_pred_as_referred_obj]
                fin_objs_pred_as_referred_obj = referred_objs[b] + ': '
                referred_zipped = zip(objs_pred_as_referred_obj, objs_pred_as_referred_obj_vals)
                for this_o, this_v in sorted(referred_zipped, reverse=True, key = lambda t: t[1]):
                    fin_objs_pred_as_referred_obj += this_o +  ' (' + this_v + '), '  
                objs_pred_as_referred_objs.append(fin_objs_pred_as_referred_obj)

                objs_pred_as_reference_obj_vals = [str(round(float(objs_pred_as_reference_obj_orig[o_i].cpu().numpy()), 4)) for o_i in objs_pred_as_reference_obj]
                objs_pred_as_reference_obj = [this_obj_labels[o_i] for o_i in objs_pred_as_reference_obj]
                fin_objs_pred_as_reference_obj = reference_objs[b] + ': '
                reference_zipped = zip(objs_pred_as_reference_obj, objs_pred_as_reference_obj_vals)
                for this_o, this_v in sorted(reference_zipped, reverse=True, key = lambda t: t[1]):
                    fin_objs_pred_as_reference_obj += this_o +  ' (' + this_v + '), '  
                objs_pred_as_reference_objs.append(fin_objs_pred_as_reference_obj)

            outputs['objs_pred_as_referred_obj'] = objs_pred_as_referred_objs 
            outputs['objs_pred_as_reference_obj'] = objs_pred_as_reference_objs 
        
        all_scores = []
        for concept in gdef.attribute_concepts:
            concept = concept.replace('_', ' ')
            if concept in feed_dict['class_to_idx'][0]:
                this_idx = feed_dict['class_to_idx'][0][concept]
                if this_idx == 607: # pad
                    all_scores.append(torch.zeros(all_f.size(0), all_f.size(1)).cuda())
                else:
                    this_score = all_f[:, :, this_idx]
                    all_scores.append(this_score)
            else:
                all_scores.append(torch.zeros(all_f.size(0), all_f.size(1)).cuda())
        all_scores = torch.stack(all_scores, dim=-1)
        
        accs, losses = [], []
        concepts_to_accs, concepts_to_pred_concepts = [], []
        for b in range(object_labels.size(0)):
            curr_concepts_to_accs = collections.defaultdict(list)
            curr_concepts_to_pred_concepts = collections.defaultdict(list)
            for i in range(object_labels.size(1)):
                curr_label = int(object_labels[b, i].cpu().numpy())
                curr_class = idx_to_class[curr_label]
                curr_class = curr_class.replace(' ', '_')
                if curr_class in gdef.attribute_concepts:
                    curr_class_index = gdef.attribute_concepts.index(curr_class)
                
                pred_scores_for_object = all_scores[b, i, :] 
                if curr_class != 'pad':
                    this_max_class = int(torch.argmax(pred_scores_for_object).cpu().numpy())
                    this_pred_class = gdef.attribute_concepts[this_max_class]
                    curr_concepts_to_pred_concepts[curr_class].append(this_pred_class)
                    this_acc = float(this_max_class == curr_class_index)               
                    accs.append(this_acc)
                    curr_concepts_to_accs[curr_class].append(this_acc)

                this_loss = torch.nn.CrossEntropyLoss(ignore_index=feed_dict['class_to_idx'][0]['pad'])(pred_scores_for_object, torch.tensor(curr_class_index).cuda())
                losses.append(this_loss)
            
            concepts_to_accs.append(curr_concepts_to_accs)
            concepts_to_pred_concepts.append(curr_concepts_to_pred_concepts)
        
        avg_acc = sum(accs) / len(accs)
        avg_loss = sum(losses) / len(losses)
        
        acc_for_refs, acc_for_references = [], []
        for i, (ref, reference) in enumerate(zip(referred_objs, reference_objs)):
            if ref in concepts_to_accs[i]:
                curr_acc_for_ref = concepts_to_accs[i][ref]
                curr_acc_for_ref = sum(curr_acc_for_ref) / len(curr_acc_for_ref)
                acc_for_refs.append(round(curr_acc_for_ref, 3))
            else:
                acc_for_refs.append(0.0)
            
            if reference in concepts_to_accs[i]:
                curr_acc_for_references = concepts_to_accs[i][reference]
                if len(curr_acc_for_references) == 0:
                    acc_for_references.append(None)
                else:
                    curr_acc_for_references = sum(curr_acc_for_references) / len(curr_acc_for_references)
                    acc_for_references.append(round(curr_acc_for_references, 3))
            else:
                acc_for_references.append(0.0)
        
        outputs['referred_objs'] = referred_objs
        outputs['anchor_objs'] = reference_objs
        outputs['acc_for_refs'] = acc_for_refs
        outputs['acc_for_ancs'] = acc_for_references
        outputs['concepts_to_accs'] = concepts_to_accs
        outputs['concepts_to_pred_concepts'] = concepts_to_pred_concepts
        
        if self.training and self.add_supervision:
            monitors['loss/object_cls'] = avg_loss
            
        if self.training:
            monitors['acc/object_cls'] = avg_acc
            monitors['train/acc/object_cls'] = avg_acc
        else:
            monitors['validation/acc/object_cls'] = avg_acc

        return monitors, outputs

