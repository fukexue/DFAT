import math
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
class WarmupStepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, warmup_epochs=0, warmup_lr_init=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        super(WarmupStepLR, self).__init__(optimizer, last_epoch)
        self.step_size = step_size
        self.gamma = gamma
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 计算 warmup 期间的学习率
            alpha = self.last_epoch / self.warmup_epochs
            lr = self.warmup_lr_init * (1 - alpha) + self.base_lrs[0] * alpha
        else:
            # 使用 StepLR 的学习率计算方式
            lr = self.base_lrs[0] * self.gamma ** (math.floor((self.last_epoch - self.warmup_epochs) / self.step_size))
        return [lr]