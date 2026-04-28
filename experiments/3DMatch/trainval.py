import argparse
import time

import torch.optim as optim

from geotransformer.engine import EpochBasedTrainer
from geotransformer.utils.scheduler import WarmupStepLR

from config import default_parse_args
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator


class Trainer(EpochBasedTrainer):
    def __init__(self, args, cfg):
        super().__init__(args, cfg, max_epoch=cfg.optim.max_epoch)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model, optimizer, scheduler
        model = create_model(cfg).cuda()
        model = self.register_model(model)
        optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.register_optimizer(optimizer)
        if hasattr(cfg.optim, 'lr_scheduler') and cfg.optim.lr_scheduler == 'warmupstep':
            scheduler = WarmupStepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay, warmup_epochs=cfg.optim.warmup_epochs, warmup_lr_init=cfg.optim.warmup_lr_init)
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def main():
    args, cfg = default_parse_args('train')
    trainer = Trainer(args, cfg)
    trainer.run()


if __name__ == '__main__':
    main()
