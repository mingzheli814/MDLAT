import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad
from advertorch.utils import NormalizeByChannelMeanStd

def at_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.train()
    criterion = nn.CrossEntropyLoss()
    adversary = L2PGDAttack(
        model, loss_fn=criterion, eps=128./255., nb_iter=10, eps_iter=15./255.,
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False
    )
    with ctx_noparamgrad(model):
        x_adv = adversary.perturb(x_natural, y)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    #logits = model(x_natural)

    logits_adv = model(x_adv)

    loss = F.cross_entropy(logits_adv, y)

    return loss
