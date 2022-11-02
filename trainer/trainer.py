from binascii import rlecode_hqx
import re
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from base import BaseTrainer
from model.metric import psnr, ssim
from model.model import MixModel, ResolveModel
from utils import inf_loop, MetricTracker
from model.loss import *


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        #('loss', 'High_R_I_loss', 'R_equal_loss', 'recon_loss_low', 'recon_loss_high', 'mutual_light_loss', 'mutual_low_loss', 'mutual_high_loss')
        self.train_metrics = MetricTracker(*('loss', 'High_R_I_loss', 'R_equal_loss', 'recon_loss_low', 'recon_loss_high', 'mutual_light_loss', 'mutual_low_loss', 'mutual_high_loss', 'R_I_color', 'mutual_I_R'), *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            low_input, high_input = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            low_R, low_L = self.model(low_input)
            high_R, high_L = self.model(high_input)
            low_L_3 = torch.cat((low_L, low_L, low_L), dim=1)
            high_L_3 = torch.cat((high_L, high_L, high_L), dim=1)

            High_R_I_loss = l2_loss(low_R, high_input)
            R_equal_loss = l1_loss(low_R, high_R)
            recon_loss_low = l1_loss((low_R * low_L_3), low_input)
            recon_loss_high = l1_loss((high_R * high_L_3), high_input)
            mutual_low_loss = smooth_light_input_loss(low_L, low_input)
            mutual_high_loss = smooth_light_input_loss(high_L, high_input)
            R_I_color = L_color(high_input, high_R)
            mutual_I_R = smooth_I_R(high_input, high_R)
            # mutual_light_loss = smooth_light_loss(low_L, high_L)
            # smooth_low_loss = smooth_L_R(low_L, low_R)
            # smooth_high_loss = smooth_L_R(high_L, high_R.detach())


            loss = recon_loss_high + recon_loss_low + R_equal_loss * 0.05 + mutual_low_loss * 0.15 + mutual_high_loss * 0.15 + High_R_I_loss * 0.01 + mutual_I_R * 0.5
            # loss = recon_loss_high/recon_loss_high.detach() + recon_loss_low/recon_loss_low.detach() + R_equal_loss/R_equal_loss.detach()  + mutual_light_loss/mutual_light_loss.detach() + mutual_low_loss/mutual_low_loss.detach() + mutual_high_loss/mutual_high_loss.detach() + High_R_I_loss/High_R_I_loss.detach()


            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('R_equal_loss', R_equal_loss.item())
            self.train_metrics.update('High_R_I_loss', High_R_I_loss.item())
            self.train_metrics.update('recon_loss_high', recon_loss_high.item())
            self.train_metrics.update('recon_loss_low', recon_loss_low.item())
            self.train_metrics.update('mutual_high_loss', mutual_high_loss.item())
            self.train_metrics.update('mutual_low_loss', mutual_low_loss.item())
            self.train_metrics.update('R_I_color', R_I_color.item())
            self.train_metrics.update('mutual_I_R', mutual_I_R.item())
            # self.writer.add_scalars('losses', 
            # {
            #     'R_equal_loss':R_equal_loss.item(),
            #     'High_R_I_loss':High_R_I_loss.item(),
            #     'recon_loss_high': recon_loss_high.item(),
            #     'recon_loss_low': recon_loss_low.item(),
            #     'mutual_light_loss': mutual_light_loss.item()
            # })
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input_low', make_grid(data.cpu(), nrow=4))
                self.writer.add_image('input_high', make_grid(target.cpu(), nrow=4))
                self.writer.add_image('output_high_R', make_grid(high_R.cpu(), nrow=4))
                self.writer.add_image('output_low_R', make_grid(low_R.cpu(), nrow=4))
                self.writer.add_image('output_high_L', make_grid(high_L_3.cpu(), nrow=4))
                self.writer.add_image('output_low_L', make_grid(low_L_3.cpu(), nrow=4))
                self.logger.debug('lr: {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):  # type: ignore
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')  # type: ignore
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size  # type: ignore
            total = self.data_loader.n_samples  # type: ignore
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class LightTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model: MixModel, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        # resolve_checkpoint = torch.load('saved/models/LLIE/1030_225419/checkpoint-epoch195.pth')
        # resolve_checkpoint = torch.load('saved/models/LLIE/1026_220321/checkpoint-epoch480.pth')
        # self.model.ResolveModel.load_state_dict(resolve_checkpoint['state_dict'])

        '''
        loss functions:

        ('loss', 'High_R_I_loss', 'R_equal_loss', 'recon_loss_low', 'recon_loss_high', 'mutual_light_loss', 'mutual_low_loss', 'mutual_high_loss')
        '''
        self.train_metrics = MetricTracker(*('loss', 'all_loss', 'High_R_I_loss', 'R_equal_loss', 'recon_loss_low', 'recon_loss_high', 'mutual_light_loss', 'mutual_low_loss', 'mutual_high_loss', 'L_color'), *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            low_input, high_input = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output, light_L, low_R, low_L = self.model(low_input)
            _, _, high_R, high_L = self.model(high_input)
            low_L_3 = torch.cat((low_L, low_L, low_L), dim=1)
            high_L_3 = torch.cat((high_L, high_L, high_L), dim=1)

            all_loss = l1_loss(high_input, output)

            R_equal_loss = l1_loss(low_R, high_R.detach())
            recon_loss_low = l1_loss((low_R * low_L_3), low_input)
            recon_loss_high = l1_loss((high_R * high_L_3), high_input)
            mutual_low_loss = smooth_light_input_loss(low_L, low_input)
            mutual_high_loss = smooth_light_input_loss(high_L, high_input)
            L_color = R_I_color()(high_input, high_R)
            # mutual_I_R = smooth_I_R(high_input, high_R)

            loss = all_loss + 0.01*R_equal_loss + 0.5*recon_loss_high + 0.5*recon_loss_low + 0.1*mutual_low_loss + 0.1*mutual_high_loss + L_color

            loss.backward()
            p = psnr(output=output.detach().cpu(), target=high_input.cpu())
            ss = ssim(output=output.detach().cpu(), target=high_input.cpu())
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('all_loss', all_loss.item())
            self.train_metrics.update('R_equal_loss', R_equal_loss.item())
            self.train_metrics.update('recon_loss_high', recon_loss_high.item())
            self.train_metrics.update('recon_loss_low', recon_loss_low.item())
            self.train_metrics.update('mutual_high_loss', mutual_high_loss.item())
            self.train_metrics.update('mutual_low_loss', mutual_low_loss.item())
            self.train_metrics.update('L_color', L_color.item())
            self.train_metrics.update('psnr', p)
            self.train_metrics.update('ssim', ss)
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input_low', make_grid(data.cpu(), nrow=4))
                self.writer.add_image('input_high', make_grid(target.cpu(), nrow=4))
                self.writer.add_image('output', make_grid(output.cpu(), nrow=4))
                self.writer.add_image('light_L', make_grid(light_L.cpu(), nrow=4))
                self.writer.add_image('output_R', make_grid(low_R.cpu(), nrow=4))
                self.logger.debug('lr: {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):  # type: ignore
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')  # type: ignore
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size  # type: ignore
            total = self.data_loader.n_samples  # type: ignore
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
