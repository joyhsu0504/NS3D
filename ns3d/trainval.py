#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Joy Hsu <joycj@stanford.edu>
#
# Distributed under terms of the MIT license.

import time
import os
import os.path as osp
import plotly.graph_objects as go
import numpy as np
import pickle
from nltk.tokenize import word_tokenize

import torch
import torch.backends.cudnn as cudnn
import torch.cuda as cuda

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.imp import load_source
from jacinle.utils.tqdm import tqdm_pbar
from jaclearn.mldash import MLDashClient

from jactorch.cli import escape_desc_name, ensure_path, dump_metainfo
from jactorch.cuda.copy import async_copy_to
from jactorch.train import TrainerEnv
from jactorch.utils.meta import as_float

from datasets.vocab import Vocab
from datasets.referit3d.referit3d_reader import *

logger = get_logger(__file__)

parser = JacArgumentParser(description='')
parser.add_argument('--desc', required=True, type='checked_file', metavar='FILE')
parser.add_argument('--expr', default='default', metavar='S', help='experiment name')
parser.add_argument('--config', type='kv', nargs='*', metavar='CFG', help='extra config')

# training hyperparameters
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='initial learning rate')
parser.add_argument('--iters-per-epoch', type=int, default=0, metavar='N', help='number of iterations per epoch 0=one pass of the dataset')
parser.add_argument('--acc-grad', type=int, default=1, metavar='N', help='accumulated gradient')
parser.add_argument('--validation-interval', type=int, default=1, metavar='N', help='validation inverval (epochs)')

# finetuning and snapshot
parser.add_argument('--load', type='checked_file', default=None, metavar='FILE', help='load the weights from a pretrained model')
parser.add_argument('--resume', type='checked_file', default=None, metavar='FILE', help='path to latest checkpoint')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='manual epoch number')
parser.add_argument('--save-interval', type=int, default=10, metavar='N', help='model save interval (epochs)')

# evaluation only
parser.add_argument('--evaluate', action='store_true', help='evaluate the performance of the model and exit')

# data related
parser.add_argument('--scannet-file', required=True, metavar='DIR', help='data directory')
parser.add_argument('--referit3D-file', required=True, metavar='DIR', help='data directory')
parser.add_argument('--data-workers', type=int, default=2, metavar='N', help='the num of workers that input training data')

# misc
parser.add_argument('--use-gpu', type='bool', default=True, metavar='B', help='use GPU or not')
parser.add_argument('--use-tb', type='bool', default=True, metavar='B', help='use tensorboard or not')
parser.add_argument('--debug', action='store_true', help='entering the debug mode, suppressing all logs to disk')
parser.add_argument('--embed', action='store_true', help='entering embed after initialization')
parser.add_argument('--force-gpu', action='store_true', help='force the script to use GPUs, useful when there exists on-the-ground devices')

args = parser.parse_args()

# filenames
args.series_name = 'ns3d'
args.desc_name = escape_desc_name(args.desc)
if not args.evaluate:
    args.run_name = 'trainval-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
else:
    args.run_name = 'val-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))

desc = load_source(args.desc)

if hasattr(desc, 'configs'):
    configs = desc.configs
else:
    from jacinle.config.environ_v2 import configs

if args.config is not None:
    from jacinle.config.environ_v2 import set_configs
    with set_configs():
        for c in args.config:
            c.apply(configs)

if args.use_gpu:
    nr_devs = cuda.device_count()
    if args.force_gpu and nr_devs == 0:
        nr_devs = 1
    assert nr_devs > 0, 'No GPU device available'
    args.gpus = [i for i in range(nr_devs)]
    args.gpu_parallel = (nr_devs > 1)

mldash = MLDashClient('dumps')


def main():
    # directories
    if not args.debug:
        args.dump_dir = ensure_path(osp.join('dumps', args.series_name, args.desc_name, args.expr, args.run_name))
        args.ckpt_dir = ensure_path(osp.join(args.dump_dir, 'checkpoints'))
        args.vis_dir = ensure_path(osp.join(args.dump_dir, 'visualizations'))
        args.meta_file = osp.join(args.dump_dir, 'metainfo.json')
        args.log_file = osp.join(args.dump_dir, 'log.log')
        args.meter_file = osp.join(args.dump_dir, 'meter.json')

        # Initialize the tensorboard.
        if args.use_tb:
            args.tb_dir = ensure_path(osp.join(args.dump_dir, 'tensorboard'))
        else:
            args.tb_dir = None

    if not args.debug:
        logger.critical('Writing logs to file: "{}".'.format(args.log_file))
        import jacinle
        jacinle.set_logger_output_file(args.log_file)
        jacinle.git_guard()

        logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
        with open(args.meta_file, 'w') as f:
            f.write(dump_metainfo(args=args.__dict__, configs=configs))

    if args.debug and args.use_tb:
        logger.warning('Disabling the tensorboard in the debug mode.')
        args.use_tb = False
    if args.evaluate and args.use_tb:
        logger.warning('Disabling the tensorboard in the evaluation mode.')
        args.use_tb = False
    
    from datasets.definition import set_global_definition
    from datasets.referit3d import ReferIt3DDefinition
    set_global_definition(ReferIt3DDefinition())

    logger.critical('Building the model.')
    model = desc.make_model(args)

    if args.use_gpu:
        model.cuda()
        # Use the customized data parallel if applicable.
        if args.gpu_parallel:
            from jactorch.parallel import JacDataParallel
            # Set user_scattered because we will add a multi GPU wrapper to the dataloader. See below.
            model = JacDataParallel(model, device_ids=args.gpus, user_scattered=True).cuda()
        # Disable the cudnn benchmark.
        cudnn.benchmark = False

    if hasattr(desc, 'make_optimizer'):
        logger.critical('Building customized optimizer.')
        optimizer = desc.make_optimizer(model, args.lr)
    else:
        from jactorch.optim import AdamW
        trainable_parameters = filter(lambda x: x.requires_grad, model.parameters())
        optimizer = AdamW(trainable_parameters, args.lr, weight_decay=configs.train.weight_decay)

    if args.acc_grad > 1:
        from jactorch.optim import AccumGrad
        optimizer = AccumGrad(optimizer, args.acc_grad)
        logger.warning('Use accumulated grad={:d}, effective iterations per epoch={:d}.'.format(args.acc_grad, int(args.iters_per_epoch / args.acc_grad)))
    

    trainer = TrainerEnv(model, optimizer)

    if args.resume:
        extra = trainer.load_checkpoint(args.resume)
        if extra:
            args.start_epoch = extra['epoch']
            logger.critical('Resume from epoch {}.'.format(args.start_epoch))
    elif args.load:
        if trainer.load_weights(args.load):
            logger.critical('Loaded weights from pretrained model: "{}".'.format(args.load))

    if args.use_tb:
        from jactorch.train.tb import TBLogger, TBGroupMeters
        tb_logger = TBLogger(args.tb_dir)
        meters = TBGroupMeters(tb_logger)
        logger.critical('Writing tensorboard logs to: "{}".'.format(args.tb_dir))
    else:
        from jacinle.utils.meter import GroupMeters
        meters = GroupMeters()

    if not args.debug:
        logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
        with open(args.meta_file, 'w') as f:
            f.write(dump_metainfo(args=args.__dict__, configs=configs))
        logger.critical('Writing meter logs to file: "{}".'.format(args.meter_file))

        logger.critical('Initializing MLDash.')
        mldash.init(
            desc_name=args.series_name + '/' + args.desc_name,
            expr_name=args.expr,
            run_name=args.run_name,
            args=args,
            highlight_args=parser,
            configs=configs,
        )
        mldash.update(metainfo_file=args.meta_file, log_file=args.log_file, meter_file=args.meter_file, tb_dir=args.tb_dir)

    if args.embed:
        from IPython import embed; embed()

    if hasattr(desc, 'customize_trainer'):
        desc.customize_trainer(trainer)

    logger.critical('Building the data loader.')
    
    from datasets.referit3d.definition import ReferIt3DDatasetSplit
    from datasets.referit3d.arguments import parse_arguments
    from datasets.referit3d.listening_dataset import make_data_loaders
    referit3d_args = parse_arguments(['-scannet-file', args.scannet_file, '-referit3D-file', args.referit3D_file, '--max-distractors', '9', '--max-test-objects', '88', '--batch-size', '16', '--n-workers', '2'])
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(referit3d_args.scannet_file)
    referit_data = load_referential_data(referit3d_args, referit3d_args.referit3D_file, scans_split)
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, referit3d_args)
    data_loaders = make_data_loaders(referit3d_args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)
    train_dataloader = data_loaders['train']
    validation_dataloader = data_loaders['test']

    if args.use_gpu and args.gpu_parallel:
        from jactorch.data.dataloader import JacDataLoaderMultiGPUWrapper
        train_dataloader = JacDataLoaderMultiGPUWrapper(train_dataloader, args.gpus)
        validation_dataloader = JacDataLoaderMultiGPUWrapper(validation_dataloader, args.gpus)

    if args.evaluate:
        epoch = 0

        model.eval()
        validate_epoch(epoch, trainer, validation_dataloader, meters, all_scans_in_dict)

        if not args.debug:
            meters.dump(args.meter_file)

        logger.critical(meters.format_simple('Epoch = {}'.format(epoch), compressed=False))
        return

    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        meters.reset()

        model.train()
        train_epoch(epoch, trainer, train_dataloader, meters, all_scans_in_dict)

        if args.validation_interval > 0 and epoch % args.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                validate_epoch(epoch, trainer, validation_dataloader, meters, all_scans_in_dict)

        if not args.debug:
            meters.dump(args.meter_file)

        if not args.debug:
            mldash.log_metric('epoch', epoch, desc=False, expr=False)
            for key, value in meters.items():
                if key.startswith('loss') or key.startswith('validation/loss'):
                    mldash.log_metric_min(key, value.avg)
            for key, value in meters.items():
                if key.startswith('acc') or key.startswith('validation/acc') or key.startswith('train/acc'):
                    mldash.log_metric_max(key, value.avg)

        logger.critical(meters.format_simple('Epoch = {}'.format(epoch), compressed=False))

        if not args.debug:
            if epoch % args.save_interval == 0:
                fname = osp.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch))
                trainer.save_checkpoint(fname, dict(epoch=epoch, meta_file=args.meta_file))


def train_epoch(epoch, trainer, train_dataloader, meters, all_scans_in_dict):
    nr_iters = args.iters_per_epoch
    if nr_iters == 0:
        nr_iters = len(train_dataloader)

    meters.update(epoch=epoch)

    trainer.trigger_event('epoch:before', trainer, epoch)
    train_iter = iter(train_dataloader)
    
    referit3dnet_class_to_idx = pickle.load(open('datasets/referit3d/data/referit3dnet_class_to_idx.p', 'rb'))
    end = time.time()
    with tqdm_pbar(total=nr_iters) as pbar:
        for i in range(nr_iters):
            feed_dict = next(train_iter)
            
            feed_dict['input_str'] = feed_dict['utterance']
            tokenized = []
            for u in feed_dict['utterance']:
                tokenized.append(word_tokenize(u))
            feed_dict['input_str_tokenized'] = tokenized
            feed_dict['input_objects'] = feed_dict['objects']
            feed_dict['input_objects_class'] = feed_dict['class_labels']
            feed_dict['input_objects_length'] = feed_dict['context_size']
            feed_dict['output_target'] = feed_dict['target_pos']
            feed_dict['class_to_idx'] = [referit3dnet_class_to_idx] * feed_dict['input_objects_class'].size(0)
            feed_dict['scene'] = None
            
            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()

            loss, monitors, output_dict, extra_info = trainer.step(feed_dict)
            step_time = time.time() - end; end = time.time()

            meters.update(loss=loss)
            meters.update(monitors)
            meters.update({'time/data': data_time, 'time/step': step_time})
            
            target = feed_dict['output_target']
            executions = output_dict['executions']
            predictions = []
            for i in range(len(executions)):
                predictions.append(torch.argmax(executions[i]))
            predictions = torch.stack(predictions)
            guessed_correctly = torch.mean((predictions == target).double()).item()
            meters.update({'train/acc': guessed_correctly})
            
            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple(
                'Epoch {}'.format(epoch),
                {k: v for k, v in meters.val.items() if not k.startswith('validation') and k.count('/') <= 1},
                compressed=True
            ), refresh=False)
            pbar.update()

            end = time.time()

    trainer.trigger_event('epoch:after', trainer, epoch)


def decode_stimulus_string(s):
        """
        Split into scene_id, instance_label, # objects, target object id,
        distractors object id.
        :param s: the stimulus string
        """
        if len(s.split('-', maxsplit=4)) == 4:
            scene_id, instance_label, n_objects, target_id = \
                s.split('-', maxsplit=4)
            distractors_ids = ""
        else:
            scene_id, instance_label, n_objects, target_id, distractors_ids = \
                s.split('-', maxsplit=4)

        instance_label = instance_label.replace('_', ' ')
        n_objects = int(n_objects)
        target_id = int(target_id)
        distractors_ids = [int(i) for i in distractors_ids.split('-') if i != '']
        assert len(distractors_ids) == n_objects - 1

        return scene_id, instance_label, n_objects, target_id, distractors_ids   
    
def validate_epoch(epoch, trainer, val_dataloader, meters, all_scans_in_dict):
    
    if not args.debug:
        from jaclearn.visualize.html_table import HTMLTableVisualizer, HTMLTableColumnDesc
        vis = HTMLTableVisualizer(osp.join(args.vis_dir, f'episode_{epoch}'), f'ReferIt3D @ Epoch {epoch}')
        link = '<a href="viewer://{}", target="_blank">{}</a>'.format(vis.visdir, vis.visdir)
        columns = [
            HTMLTableColumnDesc('id', 'Index', 'text', {'width': '40px'}),
            HTMLTableColumnDesc('utterance', 'Utterance', 'code', {'width': '500px'}),
            HTMLTableColumnDesc('referred_obj_acc', 'Referred Object Accuracy', 'text', {'width': '120px'}),
            HTMLTableColumnDesc('anchor_obj_acc', 'Anchor Object Accuracy', 'text', {'width': '120px'}),
            HTMLTableColumnDesc('object_classification_listed', 'All Objects Accuracy', 'text', {'width': '150px'}),
            HTMLTableColumnDesc('object_classification_pred', 'All Objects Preds', 'text', {'width': '200px'}),
            HTMLTableColumnDesc('preds_for_referred_obj', 'Preds for Referred Object', 'text', {'width': '200px'}),
            HTMLTableColumnDesc('preds_for_anchor_obj', 'Preds for Anchor Object', 'text', {'width': '200px'}),
            HTMLTableColumnDesc('correctness', 'Accurate', 'text', {'width': '40px'}),
        ]
        
        curr_count, max_log_count = 0, 20
    
    referit3dnet_class_to_idx = pickle.load(open('datasets/referit3d/data/referit3dnet_class_to_idx.p', 'rb'))
    end = time.time()
    with tqdm_pbar(total=len(val_dataloader)) as pbar:
        if not args.debug:
            vis.begin_html()
            vis.begin_table('ReferIt3D', columns)
            
        accuracy = []
        easy_acc, hard_acc, view_dep_acc, view_indep_acc, multi_relate_acc, non_multi_relate_acc, high_arity_acc, non_high_arity_acc = [], [], [], [], [], [], [], []
        time_list = []
        view_dependent_words = {'facing', 'looking', 'front', 'behind', 'back', 'right', 'left', 'leftmost', 'rightmost', 'across'}
        multi_relate_words = {'between', 'center', 'middle'}
        high_arity_words = {'between', 'center', 'middle', 'facing', 'looking'}
        
        for feed_dict in val_dataloader:
            
            feed_dict['input_str'] = feed_dict['utterance']
            tokenized = []
            for u in feed_dict['utterance']:
                tokenized.append(word_tokenize(u))
            feed_dict['input_str_tokenized'] = tokenized
            feed_dict['input_objects'] = feed_dict['objects']
            feed_dict['input_objects_class'] = feed_dict['class_labels']
            feed_dict['input_objects_length'] = feed_dict['context_size']
            feed_dict['output_target'] = feed_dict['target_pos']
            feed_dict['class_to_idx'] = [referit3dnet_class_to_idx] * feed_dict['input_objects_class'].size(0)
            feed_dict['scene'] = None
            
            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)
            
            data_time = time.time() - end; end = time.time()

            output_dict, extra_info = trainer.evaluate(feed_dict)           

            monitors = as_float(output_dict['monitors'])
            step_time = time.time() - end; end = time.time()

            meters.update(monitors)
            meters.update({'time/data': data_time, 'time/step': step_time})

            target = feed_dict['output_target']
            executions = output_dict['executions']
            predictions = []
            for i in range(len(executions)):           
                predictions.append(torch.argmax(executions[i]))
                
            for i in range(len(executions)):
                this_tokens = feed_dict['utterance'][i].split(' ')
                this_stimulus_id = feed_dict['stimulus_id'][i]
                
                hardness = decode_stimulus_string(this_stimulus_id)[2]
                this_easy = hardness <= 2
                this_view_dependent = len(set(this_tokens).intersection(view_dependent_words)) > 0
                this_multi_relate = len(set(this_tokens).intersection(multi_relate_words)) > 0
                this_high_arity = len(set(this_tokens).intersection(high_arity_words)) > 0                
                this_pred_acc = predictions[i] == target[i]
                
                if this_view_dependent:  
                    view_dep_acc.append(this_pred_acc)
                else:
                    view_indep_acc.append(this_pred_acc)
                    
                if this_multi_relate:
                    multi_relate_acc.append(this_pred_acc)
                else:
                    non_multi_relate_acc.append(this_pred_acc)
                
                if this_high_arity:
                    high_arity_acc.append(this_pred_acc)
                else:
                    non_high_arity_acc.append(this_pred_acc)
                
                if this_easy:
                    easy_acc.append(this_pred_acc)
                else:
                    hard_acc.append(this_pred_acc)
            
            predictions = torch.stack(predictions)
            guessed_correctly = torch.mean((predictions == target).double()).item()
            meters.update({'validation/acc': guessed_correctly})
            accuracy.append(guessed_correctly)
            
            if not args.debug and curr_count < max_log_count:
                idx = 0
                utterance = feed_dict['input_str'][idx]
                correctness = predictions[idx].cpu() == target[idx].cpu()
                
                referred_obj = output_dict['referred_objs'][idx]
                anchor_obj = output_dict['anchor_objs'][idx]
                acc_for_ref = output_dict['acc_for_refs'][idx]
                acc_for_anc = output_dict['acc_for_ancs'][idx]
                preds_for_referred_obj = output_dict['objs_pred_as_referred_obj'][idx]
                preds_for_anchor_obj = output_dict['objs_pred_as_reference_obj'][idx]
                ref = referred_obj + ': ' + str(acc_for_ref)
                anc = anchor_obj + ': ' + str(acc_for_anc)
                concepts_to_accs = str(output_dict['concepts_to_accs'][idx])
                concepts_to_pred_concepts = str(output_dict['concepts_to_pred_concepts'][idx])
                                    
                vis.row(id=curr_count, utterance=utterance, referred_obj_acc=ref, anchor_obj_acc=anc, object_classification_listed=concepts_to_accs, object_classification_pred=concepts_to_pred_concepts, preds_for_referred_obj=preds_for_referred_obj, preds_for_anchor_obj=preds_for_anchor_obj, correctness=correctness.cpu().numpy())
                curr_count += 1
            

            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple(
                'Epoch {} (validation)'.format(epoch),
                {k: v for k, v in meters.val.items() if k.startswith('validation') and k.count('/') <= 2},
                compressed=True
            ), refresh=False)
            pbar.update()

            end = time.time()
              
        monitors['validation/acc/referit3d_full'] = sum(accuracy) / len(accuracy)
        monitors['validation/acc/view_dep_acc'] = float(sum(view_dep_acc) / float(len(view_dep_acc)))
        monitors['validation/acc/view_indep_acc'] = float(sum(view_indep_acc) / float(len(view_indep_acc)))
        monitors['validation/acc/easy_acc'] = float(sum(easy_acc) / float(len(easy_acc)))
        monitors['validation/acc/hard_acc'] = float(sum(hard_acc) / float(len(hard_acc)))
        monitors['validation/acc/multi_relate_acc'] = float(sum(multi_relate_acc) / float(len(multi_relate_acc)))
        monitors['validation/acc/non_multi_relate_acc'] = float(sum(non_multi_relate_acc) / float(len(non_multi_relate_acc)))
        monitors['validation/acc/high_arity_acc'] = float(sum(high_arity_acc) / float(len(high_arity_acc)))
        monitors['validation/acc/non_high_arity_acc'] = float(sum(non_high_arity_acc) / float(len(non_high_arity_acc)))
        monitors['validation/acc/view_dep_acc_len'] = float(len(view_dep_acc))
        monitors['validation/acc/view_indep_acc_len'] = float(len(view_indep_acc))
        monitors['validation/acc/multi_relate_acc_len'] = float(len(multi_relate_acc))
        monitors['validation/acc/non_multi_relate_acc_len'] = float(len(non_multi_relate_acc))
        monitors['validation/acc/high_arity_acc_len'] = float(len(high_arity_acc))
        monitors['validation/acc/non_high_arity_acc_len'] = float(len(non_high_arity_acc))

        meters.update(monitors)
        
        if not args.debug:
            vis.end_table()
            vis.end_html()
            
        if not args.debug:
            if args.evaluate:
                mldash.update(run_description=link)
            with mldash.update_extra_info():
                mldash.extra_info_dict.setdefault('visualizations', []).append(f'Epoch {epoch:3} Visualizations: {link}')
            logger.critical(f'Visualizations: {link}')


if __name__ == '__main__':
    main()