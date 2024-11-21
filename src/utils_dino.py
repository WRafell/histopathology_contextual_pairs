'''
Mostly copy-paste from:
https://github.com/kwea123/dino_pl/blob/master/train.py
'''
from src.ssl_methods.dino import MultiCropWrapper, DINOHead, DINOLoss
from src.utils_vit import vits_dict
from omegaconf import OmegaConf
from typing import Callable
from matplotlib import cm
import torch
import numpy as np
import lightning as L


def get_dino_components(
        num_epochs: int,
        architecture: str = 'small',
        patch_size: int = 16,
        drop_path_rate: float = 0.1,
        out_dim: int = 65536,
        norm_last_layer: bool = False,
        local_crops_number: int = 8,
        warmup_teacher_temp: float = 0.04,
        final_teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 30,
        pretrained: bool = False,) -> dict:

    model = vits_dict[architecture]
    student_backbone = model(
        patch_size=patch_size,
        drop_path_rate=drop_path_rate)
    teacher_backbone = model(patch_size=patch_size)

    if pretrained:
        # Using Pretrained backbone from Lunit
        print("Pretrained ViT small backbone")
        pretrained_url = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/dino_vit_small_patch16_ep200.torch"
        state_dict = torch.hub.load_state_dict_from_url(url=pretrained_url, progress=False)
        teacher_backbone.load_state_dict(state_dict)
        student_backbone.load_state_dict(state_dict)

    student_head = DINOHead(
        student_backbone.embed_dim, 
        out_dim,
        norm_last_layer)
    teacher_head = DINOHead(teacher_backbone.embed_dim, out_dim)
    student = MultiCropWrapper(student_backbone, student_head)
    teacher = MultiCropWrapper(teacher_backbone, teacher_head)
    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters(): 
        p.requires_grad = False

    loss_function = DINOLoss(
        out_dim=out_dim,
        ncrops=local_crops_number+2,
        warmup_teacher_temp=warmup_teacher_temp,
        final_teacher_temp=final_teacher_temp,
        warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
        nepochs=num_epochs)
    
    return student, teacher, loss_function


def att2img(att, cmap=cm.get_cmap('plasma')):
    """
    att: (H, W)
    """
    x = att.cpu().numpy()
    mi, ma = np.min(x), np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x_ = cmap(x)[..., :3] # (H, W, 3)
    x_ = torch.from_numpy(x_).permute(2, 0, 1) # (3, H, W)
    return x_


def cosine_scheduler(
        base_value, 
        final_value, 
        epochs, 
        niter_per_ep,
        warmup_epochs: int = 0, 
        start_warmup_value: float = 1e-6):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * \
                             (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class DINOSSLModule(L.LightningModule):
    def __init__(
            self, 
            positive_pairs_sampling_method: Callable,
            saving_path: str,
            hparams: OmegaConf):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.automatic_optimization = False

        self.positive_pairs_sampling_method = positive_pairs_sampling_method
        self.saving_path = saving_path 
        self.temp_step_losses = []
        self.step_losses = []
        self.epoch_losses = []

        self.hparams = hparams
        dino_components = get_dino_components(hparams)
        self.teacher = dino_components['teacher']
        self.student = dino_components['student']
        self.loss_function = dino_components['loss_function']

    def configure_optimizers(self):
        regularized, not_regularized = [], []
        for n, p in self.student.named_parameters():
            if not p.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if n.endswith(".bias") or len(p.shape) == 1:
                not_regularized.append(p)
            else:
                regularized.append(p)
        param_groups = [{'params': regularized},
                        {'params': not_regularized, 'weight_decay': 0.}]

        self.lr = self.hparams.lr * (self.hparams.batch_size * self.hparams.num_gpus/256)
        opt = torch.optim.AdamW(param_groups, self.lr)

        self.lr_sch = cosine_scheduler(
            self.lr, 1e-6, 
            self.hparams.num_epochs, 
            self.hparams.niter_per_ep//self.hparams.num_gpus,
            self.hparams.warmup_epochs)
        # weight decay scheduler
        self.wd_sch = cosine_scheduler(
            self.hparams.weight_decay_init, 
            self.hparams.weight_decay_end,
            self.hparams.num_epochs, 
            self.hparams.niter_per_ep//self.hparams.num_gpus)
        # momentum scheduler
        self.mm_sch = cosine_scheduler(
            self.hparams.momentum_teacher, 1.0,
            self.hparams.num_epochs, 
            self.hparams.niter_per_ep//self.hparams.num_gpus)

        return opt
    
    def training_step(self, batch, batch_idx):

        global_views, local_views = self.positive_pairs_sampling_method(batch)

        opt = self.optimizers()
        # update learning rate, weight decay
        for i, param_group in enumerate(opt.param_groups):
            param_group['lr'] = self.lr_sch[self.global_step]
            if i == 0: # only the first group is regularized
                param_group['weight_decay'] = self.wd_sch[self.global_step]

        teacher_output = self.teacher(batch[:2])
        student_output = self.student(batch)
        loss = self.loss(student_output, teacher_output, self.current_epoch)

        opt.zero_grad()
        self.manual_backward(loss)
        # clip gradient
        if self.hparams.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.hparams.clip_grad)
        # cancel gradient for the first epochs
        if self.current_epoch < self.hparams.ep_freeze_last_layer:
            for n, p in self.student.named_parameters():
                if "last_layer" in n:
                    p.grad = None
        opt.step()

        # EMA update for the teacher
        m = self.mm_sch[self.global_step]
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_((1-m)*ps.data)










