import os
import sys
import warnings
import copy
import json
import shutil

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm as tq
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import wandb
from omegaconf import OmegaConf

import models
import dataloaders
import losses


class Runner:

    def __init__(self, config, device):
        self._config = config
        self._device = device

    def _restore(self, model, optimizer, scheduler, which='last', epoch=None):
        # BEST EPOCH IS DEFINED IN TERMS OF TASK 1 VAL MACRO F1 ONLY.

        loop_state = torch.load(
            os.path.join(self._config.training.out_base_path,
                         self._config.training.run_name, 'logs',
                         'loop_state.pt'))

        # FIGURE OUT WHICH EPOCH TO RESTORE.

        if which == 'last':
            epoch_to_get = loop_state['last_epoch']

        elif which == 'best':
            epoch_to_get = loop_state['best_epoch']

        elif which == 'epoch':
            if epoch is None:
                raise ValueError(
                    'Argument \'epoch\' cannot be None if \'which\' is \'epoch\'.'
                )
            epoch_to_get = epoch

        else:
            raise ValueError(
                'Argument \'which\' must be one of \'last\', \'best\', \'epoch\'.'
            )

        # RESTORE EPOCH.

        checkpoint = torch.load(
            os.path.join(self._config.training.out_base_path,
                         self._config.training.run_name, 'weights',
                         f'checkpoint_{epoch_to_get}.pt'))

        print(f'Restoring from epoch: {epoch_to_get} (criterion: {which}).',
              file=sys.stdout,
              flush=True)
        warnings.warn(
            'Restoring from checkpoint does not restore the config. Manually ensure the configurations of the two runs are compatible.'
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return {
            'best_epoch': loop_state['best_epoch'],
            'best_val_loss': loop_state['best_val_loss'],
            'best_val_accuracy': loop_state['best_val_accuracy'],
            'best_val_macro_f1': loop_state['best_val_macro_f1'],
            'best_val_precision': loop_state['best_val_precision'],
            'best_val_recall': loop_state['best_val_recall'],
            #
            'checkpoint_epoch': checkpoint['epoch'],
            #
            'checkpoint_val_loss': checkpoint['val_loss'],
            'checkpoint_val_accuracy': checkpoint['val_accuracy'],
            'checkpoint_val_macro_f1': checkpoint['val_macro_f1'],
            'checkpoint_val_precision': checkpoint['val_precision'],
            'checkpoint_val_recall': checkpoint['val_recall'],
            #
            'checkpoint_train_loss': checkpoint['train_loss'],
            'checkpoint_train_accuracy': checkpoint['train_accuracy'],
            'checkpoint_train_macro_f1': checkpoint['train_macro_f1'],
            'checkpoint_train_precision': checkpoint['train_precision'],
            'checkpoint_train_recall': checkpoint['train_recall'],
        }

    def _log_epoch(self, train_metrics, val_metrics, best_val_metrics, model,
                   optimizer, scheduler, epoch):
        # TODO(jiviteshjain): Add confusion matrix plotting.

        out_path = os.path.join(self._config.training.out_base_path,
                                self._config.training.run_name)

        # CHECK IF IMPROVED VAL PERFORMANCE.

        improved = val_metrics['macro_f1'] > best_val_metrics[
            'macro_f1']  # Always true on the first epoch.
        if improved:
            best_val_metrics = copy.deepcopy(val_metrics)

        # LOG TO WANDB.

        if self._config.training.wandb.use:
            wandb.log({
                'task1': {
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                },
            })

            if improved:
                for k, v in best_val_metrics.items():
                    wandb.run.summary[k] = v
                wandb.run.summary['best_epoch'] = epoch

        # SAVE CHECKPOINT.
        if self._config.training.save_weights:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
            }
            for k, v in train_metrics.items():
                checkpoint[f'train_{k}'] = v
            for k, v in val_metrics.items():
                checkpoint[f'val_{k}'] = v

            torch.save(
                checkpoint,
                os.path.join(out_path, 'weights', f'checkpoint_{epoch}.pt'))

        # UPDATE LOOP STATE.

        loop_state_path = os.path.join(out_path, 'logs', 'loop_state.pt')
        if os.path.exists(loop_state_path):
            loop_state = torch.load(loop_state_path)
        else:
            loop_state = {}

        loop_state['last_epoch'] = epoch

        if improved:  # Always true on the first epoch.
            loop_state['best_epoch'] = epoch
            for k, v in best_val_metrics.items():
                loop_state[f'best_val_{k}'] = v

        torch.save(loop_state, loop_state_path)

        # UPDATE RUN FILES

        run_path = os.path.join(out_path, 'logs', 'run.json')
        if os.path.exists(run_path):
            with open(run_path, 'r') as f:
                runs = json.load(f)
        else:
            runs = []

        runs.append({
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        })

        with open(run_path, 'w') as f:
            json.dump(runs, f, indent=2)

        return improved, best_val_metrics

    def _train_step(self, model, train_data, loss_function, optimizer, epoch):
        epoch_loss = 0
        num_batches = 0
        epoch_gt = []
        epoch_pred = []

        model.train()

        for batch in tq(train_data, desc=f'Train epoch: {epoch}'):
            targets = batch['target'].to(self._device, dtype=torch.long)
            outputs = model(batch)

            loss = loss_function(outputs, targets)
            epoch_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)

            epoch_pred.extend(predictions.tolist())
            epoch_gt.extend(targets.tolist())
            num_batches += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        accuracy = accuracy_score(epoch_gt, epoch_pred)
        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            epoch_gt, epoch_pred, average='macro')
        epoch_loss /= num_batches

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'macro_f1': macro_f1,
            'loss': epoch_loss,
        }

    def _val_step(self, model, val_data, loss_function, epoch):
        epoch_loss = 0
        num_batches = 0
        epoch_gt = []
        epoch_pred = []

        model.eval()
        with torch.no_grad():
            for batch in tq(val_data, desc=f'Val epoch: {epoch}'):
                targets = batch['target'].to(self._device, dtype=torch.long)
                outputs = model(batch)

                loss = loss_function(outputs, targets)
                epoch_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)

                epoch_pred.extend(predictions.tolist())
                epoch_gt.extend(targets.tolist())
                num_batches += 1

        accuracy = accuracy_score(epoch_gt, epoch_pred)
        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            epoch_gt, epoch_pred, average='macro')
        epoch_loss /= num_batches

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'macro_f1': macro_f1,
            'loss': epoch_loss,
        }

    def train(self, restore=False):
        # GET MODEL, DATASETS, DATALOADERS.

        model, tokenizer, max_length = getattr(
            models, self._config.training.model_in_use).get_model(
                self._config, self._device)

        train_set, val_set, _ = getattr(
            dataloaders, self._config.training.dataset_in_use).get_dataset(
                self._config, tokenizer, max_length)

        train_data = DataLoader(
            train_set,
            batch_size=self._config.training.train_batch_size,
            shuffle=self._config.training.train_shuffle,
            num_workers=self._config.training.num_workers)

        val_data = DataLoader(val_set,
                              batch_size=self._config.training.val_batch_size,
                              shuffle=self._config.training.val_shuffle,
                              num_workers=self._config.training.num_workers)

        # GET LOSS FUNCTION AND OPTIMIZER.

        loss_function = getattr(losses, self._config.training.loss_in_use)(self._config,
                               self._device)

        # Optimizer isn't configurable because who tf changes it.
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=self._config.training.learning_rate,
            weight_decay=self._config.training.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self._config.training.lr_decay_epochs,
            gamma=self._config.training.lr_decay_factor)

        # RESTORE CHECKPOINT OR COLD START.

        if restore:
            state = self._restore(model, optimizer, scheduler, which='last')
            best_val_metrics = {
                'accuracy': state['best_val_accuracy'],
                'precision': state['best_val_precision'],
                'recall': state['best_val_recall'],
                'macro_f1': state['best_val_macro_f1'],
                'loss': state['best_val_loss'],
            }
            current_epoch = state['checkpoint_epoch'] + 1

        else:
            best_val_metrics = {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'macro_f1': 0,
                'loss': np.inf,
            }
            current_epoch = 0

        # MAIN TRAINING LOOP, GO OVER ALL EPOCHS.

        # Avoid tdqm here, because multibar never works.
        while current_epoch < self._config.training.num_epochs:
            train_metrics = self._train_step(model, train_data, loss_function,
                                             optimizer, current_epoch)

            val_metrics = self._val_step(model, val_data, loss_function,
                                         current_epoch)

            _, best_val_metrics = self._log_epoch(train_metrics, val_metrics,
                                                  best_val_metrics, model,
                                                  optimizer, scheduler,
                                                  current_epoch)

            scheduler.step()
            current_epoch += 1


def train(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_path = os.path.join(config.training.out_base_path,
                            config.training.run_name)

    if config.training.force_cold_start and os.path.exists(out_path):
        warnings.warn(f'Deleting existing run of the same name: {out_path}.')
        shutil.rmtree(out_path)

    restore = os.path.exists(out_path)

    if restore:
        warnings.warn(
            'Config is not restored, but overwritten. Manually ensure current and previous configs are compatible.'
        )
    else:
        os.makedirs(os.path.join(out_path, 'logs'), exist_ok=False)
        os.makedirs(os.path.join(out_path, 'weights'), exist_ok=False)

    with open(os.path.join(out_path, 'logs', 'config.yaml'), 'w') as f:
        OmegaConf.save(config, f)

    if config.training.wandb.use:
        warnings.warn(
            'If reusing run-name, wandb resume is not perfect. Cross check manually.'
        )

        wandb.login()
        resume = 'allow' if restore else None
        wandb.init(name=config.training.run_name,
                   project=config.training.wandb.project,
                   entity=config.training.wandb.entity,
                   config=OmegaConf.to_container(config),
                   resume=resume)

    runner = Runner(config, device)
    runner.train(restore)


def test(config):
    raise NotImplementedError


# TODO(jiviteshjain):
# huggingface fast tokenizer
# wandb run sync
# tokenizer max len warning
# make warnings look good
