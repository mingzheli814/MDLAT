import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def kl_loss_inter(output1, output2):
    criterion_kl = nn.KLDivLoss(size_average=False)
    loss = (1.0 / output1.size(0))*criterion_kl(F.log_softmax(output1, dim=1), F.softmax(output2, dim=1)) + (1.0 / output1.size(0))*criterion_kl(F.log_softmax(output2, dim=1), F.softmax(output1, dim=1))
    return loss    

def at_loss(model1,
              model2,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    model1.eval()
    model2.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv1 = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    x_adv2 = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    delta1 = 0.001 * torch.randn(x_natural.shape).cuda().detach()
    delta1 = Variable(delta1.data, requires_grad=True)
    delta2 = 0.001 * torch.randn(x_natural.shape).cuda().detach()
    delta2 = Variable(delta2.data, requires_grad=True)

    # Setup optimizers
    optimizer_delta = torch.optim.SGD([delta1, delta2], lr=epsilon / perturb_steps * 2)

    for _ in range(perturb_steps):
        adv1 = x_natural + delta1
        adv2 = x_natural + delta2
        output1 = model1(adv1)
        output2 = model2(adv2)
        # optimize
        optimizer_delta.zero_grad()
        with torch.enable_grad():
            loss = (-1) * (F.cross_entropy(output1, y)+F.cross_entropy(output2, y)+kl_loss_inter(output1, output2))
        loss.backward()
        # renorming gradient
        grad_norms = delta1.grad.view(batch_size, -1).norm(p=2, dim=1)
        delta1.grad.div_(grad_norms.view(-1, 1, 1, 1))
        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta1.grad[grad_norms == 0] = torch.randn_like(delta1.grad[grad_norms == 0])
        grad_norms = delta2.grad.view(batch_size, -1).norm(p=2, dim=1)
        delta2.grad.div_(grad_norms.view(-1, 1, 1, 1))
        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta2.grad[grad_norms == 0] = torch.randn_like(delta2.grad[grad_norms == 0])
        optimizer_delta.step()

        # projection
        delta1.data.add_(x_natural)
        delta1.data.clamp_(0, 1).sub_(x_natural)
        delta1.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        delta2.data.add_(x_natural)
        delta2.data.clamp_(0, 1).sub_(x_natural)
        delta2.data.renorm_(p=2, dim=0, maxnorm=epsilon)
    x_adv1 = Variable(x_natural + delta1, requires_grad=False)
    x_adv2 = Variable(x_natural + delta2, requires_grad=False)
    model1.train()
    model2.train()

    x_adv1 = Variable(torch.clamp(x_adv1, 0.0, 1.0), requires_grad=False)
    x_adv2 = Variable(torch.clamp(x_adv2, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    #logits = model(x_natural)

    logits_adv1 = model1(x_adv1)
    logits_adv2 = model2(x_adv2)

    loss = F.cross_entropy(logits_adv1, y)+F.cross_entropy(logits_adv2, y)+kl_loss_inter(logits_adv1, logits_adv2)

    return loss
