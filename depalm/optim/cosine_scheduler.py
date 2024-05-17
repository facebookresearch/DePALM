import math
import torch
import logging


log = logging.getLogger(__name__)

class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    r"""
    Based on torch CosineAnnealingWarmRestarts
    https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#1336

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        cycle_length (int): Number of iterations for the first restart.
        cycle_length_mult (int, optional): A factor increases cycle_length after a restart. Default: 1.
        lr_min_fact (float, optional): Minimum learning rate factor. The minimu lr will be lr * lr_min_fact. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        warmup_fract: for how long to do the warmup (for steps <= cycle_length * warmup_fract)
    """

    def __init__(self, optimizer, cycle_length='full', cycle_length_mult=1., lr_min_fact=0.1, warmup_fract=0.1, last_epoch=-1, total_epochs=None, verbose=False):
        if cycle_length == 'full':
            cycle_length = total_epochs
        log.info(f"Scheduler cycle of length {cycle_length}")

        self.init_cycle_length = cycle_length
        self.cur_cycle_length = cycle_length
        self.init_cycle_length_mult = cycle_length_mult
        self.lr_min_fact = lr_min_fact
        self.T_cur = last_epoch
        self.warmup_fract = warmup_fract
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        warmup_length = self.cur_cycle_length * self.warmup_fract
        if self.T_cur < warmup_length: # Linear warmup
            cur_time = self.T_cur / warmup_length
            return [base_lr * (self.lr_min_fact + (1 - self.lr_min_fact) * cur_time)
                for base_lr in self.base_lrs]
        else: # Cos function
            cur_time = (self.T_cur - warmup_length) / (self.cur_cycle_length - warmup_length)
            return [base_lr * (self.lr_min_fact + (1 - self.lr_min_fact) * (1 + math.cos(math.pi * cur_time)) / 2)
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """
        Step could be called after every batch update
        Example: scheduler.step(epoch + i / iters)
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.cur_cycle_length:
                self.T_cur = self.T_cur - self.cur_cycle_length
                self.cur_cycle_length = self.cur_cycle_length * self.init_cycle_length_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.init_cycle_length:
                if self.init_cycle_length_mult == 1:
                    self.T_cur = epoch % self.init_cycle_length
                else:
                    n = int(math.log((epoch / self.init_cycle_length * (self.init_cycle_length_mult - 1) + 1), self.init_cycle_length_mult))
                    self.T_cur = epoch - self.init_cycle_length * (self.init_cycle_length_mult ** n - 1) / (self.init_cycle_length_mult - 1)
                    self.cur_cycle_length = self.init_cycle_length * self.init_cycle_length_mult ** (n)
            else:
                self.cur_cycle_length = self.init_cycle_length
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]