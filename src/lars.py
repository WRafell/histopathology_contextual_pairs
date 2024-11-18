import math
import torch
import lightning as L


class AdjustLearningRate(L.Callback):
    def __init__(self, max_epochs: int, batch_size: int, base_lr: float, num_batches: int):
        super().__init__()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.num_batches = num_batches
        self.warmup_steps = 10 * num_batches
        self.max_steps = max_epochs * num_batches
        self.base_lr_adjusted = base_lr * batch_size / 256

    def adjust_learning_rate(self, optimizer: torch.optim, step: int):
        if step < self.warmup_steps:
            lr = self.base_lr_adjusted * step / self.warmup_steps
        else:
            step -= self.warmup_steps
            max_steps_adjusted = self.max_steps - self.warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps_adjusted))
            end_lr = self.base_lr_adjusted * 0.001
            lr = self.base_lr_adjusted * q + end_lr * (1 - q)        
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr        
        return lr
    
    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch, batch_idx: int):
        # Calculate the current step
        current_step = trainer.global_step
        # Adjust the learning rate
        lr = self.adjust_learning_rate(pl_module.trainer.optimizers[0], current_step)
        # Log the learning rate for tracking
        pl_module.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)


# https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
class LARS(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr,
            weight_decay=0,
            momentum=0.9,
            eta=0.001,
            weight_decay_filter=None,
            lars_adaptation_filter=None):
        
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])

        return loss


def exclude_bias_and_norm(p):
    return p.ndim == 1