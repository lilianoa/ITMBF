import os
import time
import math
import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
import threading
from bootstrap.lib import utils
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

class Engine(object):
    '''
    Ours engine, Contains training and evaluation procedures
    '''
    def __init__(self):
        self.hooks = {}
        self.epoch = 0
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.view = None
        self.best_out = {}

    def generate_view(self):
        """ Generate a view.html via an asynchronous call to `self.view.generate()`
        """
        if self.view is not None:
            threading.Thread(target=self.view.generate).start()
        # path_opts = os.path.join(Options()['exp']['dir'], 'options.yaml')
        # os.system('python -m bootstrap.views.view --path_opts {}'.format(path_opts))

    def load_state_dict(self, state):
        """
        """
        self.epoch = state['epoch']
        self.best_out = state['best_out']

    def state_dict(self):
        """
        """
        state = {}
        state['epoch'] = self.epoch
        state['best_out'] = self.best_out
        return state

    def hook(self, name):
        """ Run all the callback functions that have been registered
            for a hook.

            Args:
                name: the name of the hook
        """
        if name in self.hooks:
            for func in self.hooks[name]:
                func()

    def register_hook(self, name, func):
        """ Register a callback function to be triggered when the hook
            is called.

            Args:
                name: the name of the hook
                func: the callback function (no argument)

            Example usage:

            .. code-block:: python

                def func():
                    print('hooked!')

                engine.register_hook('train_on_start_batch', func)
        """
        if name not in self.hooks:
            self.hooks[name] = []
        self.hooks[name].append(func)

    def resume(self, map_location=None):
        """ Resume a checkpoint using the `bootstrap.lib.options.Options`
        """
        Logger()('Loading {} checkpoint'.format(Options()['exp']['resume']))
        self.load(Options()['exp']['dir'],
                  Options()['exp']['resume'],
                  self.model, self.optimizer,
                  map_location=map_location)
        self.epoch += 1

    def eval(self):
        """ Launch evaluation procedures
        """
        Logger()('Launching evaluation procedures')

        if Options()['dataset']['eval_split']:
            # self.epoch-1 to be equal to the same resumed epoch
            # or to be equal to -1 when not resumed
            self.evaluate_epoch(self.model, self.dataset['eval'], self.epoch-1, logs_json=True)
        Logger()('Ending evaluation procedures')

    def train(self):
        Logger()('Launching training procedure')
        lr_default = 1e-3 if self.dataset['eval'] is not None else 7e-4
        lr_decay_step = 2
        lr_decay_rate = .25
        lr_decay_epochs = range(10, 20, lr_decay_step) if self.dataset['eval'] is not None else range(10, 20, lr_decay_step)
        gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
        grad_clip = .25
        self.optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr_default)
        Logger()('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
         (lr_default, lr_decay_step, lr_decay_rate, grad_clip))
        while self.epoch < Options()['engine']['nb_epochs']:
            self.train_epoch(self.model, self.dataset['train'], self.optimizer, self.epoch, lr_decay_rate, lr_decay_epochs, gradual_warmup_steps)

            if Options()['dataset']['eval_split']:
                out = self.evaluate_epoch(self.model, self.dataset['eval'], self.epoch)
                if 'saving_criteria' in Options()['engine'] and Options()['engine']['saving_criteria'] is not None:
                    for saving_criteria in Options()['engine']['saving_criteria']:
                        if self.is_best(out, saving_criteria):
                            name = saving_criteria.split(':')[0]
                            Logger()('Saving best checkpoint for strategy {}'.format(name))
                            self.save(Options()['exp']['dir'], 'best_{}'.format(name), self.model, self.optimizer)

            Logger()('Saving last checkpoint')
            self.save(Options()['exp']['dir'], 'last', self.model, self.optimizer)
            self.epoch += 1
        Logger()('Ending training procedures')


    def train_epoch(self, model, dataset, optim, epoch, lr_decay_rate, lr_decay_epochs, gradual_warmup_steps, mode='train'):
        utils.set_random_seed(Options()['misc']['seed'] + epoch)  #  to be able to reproduce exps on reload
        Logger()('Training model on {}set for epoch {}'.format(dataset.split, epoch))
        timer = {
            'begin': time.time(),
            'elapsed': time.time(),
            'process': None,
            'load': None,
            'run_avg': 0
        }
        out_epoch = {}
        batch_loader = DataLoader(
            dataset=dataset,
            batch_size=Options()['dataset.batch_size'],
            shuffle=True,
            num_workers=Options()['dataset.nb_threads'],
            pin_memory=Options()['misc.cuda']
        )
        if epoch < len(gradual_warmup_steps):
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            Logger()('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            Logger()('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        else:
            Logger()('lr: %.4f' % optim.param_groups[0]['lr'])

        model.train()
        for i, batch in enumerate(batch_loader):
            timer['load'] = time.time() - timer['elapsed']
            optim.zero_grad()
            # forward
            out = model(batch)
            # backward
            if not torch.isnan(out['loss']):
                out['loss'].backward()
            else:
                Logger()('NaN detected')
            # update parameter
            optim.step()
            timer['process'] = time.time() - timer['elapsed']
            if i == 0:
                timer['run_avg'] = timer['process']
            else:
                timer['run_avg'] = timer['run_avg'] * 0.8 + timer['process'] * 0.2

            Logger().log_value(f'{mode}_batch.epoch', epoch, should_print=False)
            Logger().log_value(f'{mode}_batch.batch', i, should_print=False)
            Logger().log_value(f'{mode}_batch.timer.process', timer['process'], should_print=False)
            Logger().log_value(f'{mode}_batch.timer.load', timer['load'], should_print=False)

            for key, value in out.items():
                if torch.is_tensor(value):
                    if value.dim() <= 1:
                        value = value.item() # get number from a torch scalar
                    else:
                        continue
                if type(value) == list:
                    continue
                if type(value) == dict:
                    continue
                if key not in out_epoch:
                    out_epoch[key] = []
                out_epoch[key].append(value)
                Logger().log_value(f'{mode}_batch.'+key, value, should_print=False)

            if i % Options()['engine']['print_freq'] == 0:
                Logger()("{}: epoch {} | batch {}/{}".format(mode, epoch, i, len(batch_loader) - 1))
                Logger()("{} elapsed: {} | left: {}".format(' '*len(mode),
                    datetime.timedelta(seconds=math.floor(time.time() - timer['begin'])),
                    datetime.timedelta(seconds=math.floor(timer['run_avg'] * (len(batch_loader) - 1 - i)))))
                Logger()("{} process: {:.5f} | load: {:.5f}".format(' '*len(mode), timer['process'], timer['load']))
                Logger()("{} loss: {:.5f}".format(' '*len(mode), out['loss'].data.item()))

            timer['elapsed'] = time.time()

        Logger().log_value(f'{mode}_epoch.epoch', epoch, should_print=True)
        for key, value in out_epoch.items():
            Logger().log_value(f'{mode}_epoch.'+key, sum(value)/len(value), should_print=True)
        Logger().flush()
        self.hook(f'{mode}_on_flush')


    def evaluate_epoch(self, model, dataset, epoch, mode='eval', logs_json=True):
        """ Launch evaluation procedures for one epoch
            Returns:
                out(dict): mean of all the scalar outputs of the model, indexed by output name, for this epoch
        """
        utils.set_random_seed(Options()['misc']['seed'] + epoch)  #  to be able to reproduce exps on reload
        Logger()('Evaluating model on {}set for epoch {}'.format(dataset.split, epoch))
        model.eval()

        timer = {
            'begin': time.time(),
            'elapsed': time.time(),
            'process': None,
            'load': None,
            'run_avg': 0
        }
        out_epoch = {}
        batch_loader = DataLoader(
            dataset=dataset,
            batch_size=Options()['dataset.batch_size'],
            shuffle=True,
            num_workers=Options()['dataset.nb_threads'],
            pin_memory=Options()['misc.cuda']
        )
        for i, batch in enumerate(batch_loader):
            timer['load'] = time.time() - timer['elapsed']

            with torch.no_grad():
                out = model(batch)

            timer['process'] = time.time() - timer['elapsed']
            if i == 0:
                timer['run_avg'] = timer['process']
            else:
                timer['run_avg'] = timer['run_avg'] * 0.8 + timer['process'] * 0.2

            Logger().log_value('{}_batch.batch'.format(mode), i, should_print=False)
            Logger().log_value('{}_batch.epoch'.format(mode), epoch, should_print=False)
            Logger().log_value('{}_batch.timer.process'.format(mode), timer['process'], should_print=False)
            Logger().log_value('{}_batch.timer.load'.format(mode), timer['load'], should_print=False)

            for key, value in out.items():
                if torch.is_tensor(value):
                    if value.dim() <= 1:
                        value = value.item() # get number from a torch scalar
                    else:
                        continue
                if type(value) == list:
                    continue
                if type(value) == dict:
                    continue
                if key not in out_epoch:
                    out_epoch[key] = []
                out_epoch[key].append(value)
                Logger().log_value('{}_batch.{}'.format(mode, key), value, should_print=False)

            if i % Options()['engine']['print_freq'] == 0:
                Logger()("{}: epoch {} | batch {}/{}".format(mode, epoch, i, len(batch_loader) - 1))
                Logger()("{}  elapsed: {} | left: {}".format(' ' * len(mode),
                                                             datetime.timedelta(
                                                                 seconds=math.floor(time.time() - timer['begin'])),
                                                             datetime.timedelta(seconds=math.floor(
                                                                 timer['run_avg'] * (len(batch_loader) - 1 - i)))))
                Logger()("{}  process: {:.5f} | load: {:.5f}".format(' ' * len(mode), timer['process'], timer['load']))
                self.hook('{}_on_print'.format(mode))

            timer['elapsed'] = time.time()

        out = {}
        for key, value in out_epoch.items():
            try:
                out[key] = sum(value)/len(value)
            except:
                import ipdb; ipdb.set_trace()

        Logger().log_value('{}_epoch.epoch'.format(mode), epoch, should_print=True)
        for key, value in out.items():
            Logger().log_value('{}_epoch.{}'.format(mode, key), value, should_print=True)
        if logs_json:
            Logger().flush()

        self.hook('{}_on_flush'.format(mode))
        return out

    def is_best(self, out, saving_criteria):
        """ Verify if the last model is the best for a specific saving criteria

            Args:
                out(dict): mean of all the scalar outputs of model indexed by output name
                saving_criteria(str):

            Returns:
                is_best(bool)

            Example usage:

            .. code-block:: python

                out = {
                    'loss': 0.2,
                    'acctop1': 87.02
                }

                engine.is_best(out, 'loss:min')
        """
        if ':min' in saving_criteria:
            name = saving_criteria.replace(':min', '')
            order = '<'
        elif ':max' in saving_criteria:
            name = saving_criteria.replace(':max', '')
            order = '>'
        else:
            error_msg = """'--engine.saving_criteria' named '{}' does not specify order,
            you need to chose between '{}' or '{}' to specify if the criteria needs to be minimize or maximize""".format(
                saving_criteria, saving_criteria + ':min', saving_criteria + ':max')
            raise ValueError(error_msg)

        if name not in out:
            raise KeyError("'--engine.saving_criteria' named '{}' not in outputs '{}'".format(name, list(out.keys())))

        if name not in self.best_out:
            self.best_out[name] = out[name]
        else:
            if eval('{} {} {}'.format(out[name], order, self.best_out[name])):
                self.best_out[name] = out[name]
                return True

        return False

    def load(self, dir_logs, name, model, optimizer, map_location=None):
        """ Load a checkpoint

            Args:
                dir_logs: directory of the checkpoint
                name: name of the checkpoint
                model: model associated to the checkpoint
                optimizer: optimizer associated to the checkpoint
        """
        path_template = os.path.join(dir_logs, 'ckpt_{}_{}.pth.tar')

        Logger()('Loading model...')
        model_state = torch.load(path_template.format(name, 'model'), map_location=map_location)
        model.load_state_dict(model_state)

        if Options()['dataset']['train_split'] is not None:
            if os.path.isfile(path_template.format(name, 'optimizer')):
                Logger()('Loading optimizer...')
                optimizer_state = torch.load(path_template.format(name, 'optimizer'), map_location=map_location)
                optimizer.load_state_dict(optimizer_state)
            else:
                Logger()('No optimizer checkpoint', log_level=Logger.WARNING)

        if os.path.isfile(path_template.format(name, 'engine')):
            Logger()('Loading engine...')
            engine_state = torch.load(path_template.format(name, 'engine'), map_location=map_location)
            self.load_state_dict(engine_state)
        else:
            Logger()('No engine checkpoint', log_level=Logger.WARNING)

    def save(self, dir_logs, name, model, optimizer):
        """ Save a checkpoint

            Args:
                dir_logs: directory of the checkpoint
                name: name of the checkpoint
                model: model associated to the checkpoint
                optimizer: optimizer associated to the checkpoint
        """
        path_template = os.path.join(dir_logs, 'ckpt_{}_{}.pth.tar')

        Logger()('Saving model...')
        model_state = model.state_dict()
        torch.save(model_state, path_template.format(name, 'model'))

        Logger()('Saving optimizer...')
        optimizer_state = optimizer.state_dict()
        torch.save(optimizer_state, path_template.format(name, 'optimizer'))

        Logger()('Saving engine...')
        engine_state = self.state_dict()
        torch.save(engine_state, path_template.format(name, 'engine'))



