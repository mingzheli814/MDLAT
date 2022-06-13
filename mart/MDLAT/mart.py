import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def kl_loss_inter(output1, output2):
    criterion_kl = nn.KLDivLoss(size_average=False)
    loss = (1.0 / output1.size(0))*criterion_kl(F.log_softmax(output1, dim=1), F.softmax(output2, dim=1)) + (1.0 / output1.size(0))*criterion_kl(F.log_softmax(output2, dim=1), F.softmax(output1, dim=1))
    return loss    

def kl_loss_single(output1, output2):
    criterion_kl = nn.KLDivLoss(size_average=False)
    loss = (1.0 / output1.size(0))*criterion_kl(F.log_softmax(output1, dim=1), F.softmax(output2, dim=1))
    return loss   

def mart_loss(model1, model2,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model1.eval()
    model2.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv1 = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    x_adv2 = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv1.requires_grad_()
            x_adv2.requires_grad_()
            output1 = model1(x_adv1)
            output2 = model2(x_adv2)
            with torch.enable_grad():
                loss_ce = F.cross_entropy(output1, y)+F.cross_entropy(output2, y)+2*kl_loss_inter(output1, output2)
            grad = torch.autograd.grad(loss_ce, [x_adv1, x_adv2])
            x_adv1 = x_adv1.detach() + step_size * torch.sign(grad[0].detach())
            x_adv1 = torch.min(torch.max(x_adv1, x_natural - epsilon), x_natural + epsilon)
            x_adv1 = torch.clamp(x_adv1, 0.0, 1.0)
            x_adv2 = x_adv2.detach() + step_size * torch.sign(grad[1].detach())
            x_adv2 = torch.min(torch.max(x_adv2, x_natural - epsilon), x_natural + epsilon)
            x_adv2 = torch.clamp(x_adv2, 0.0, 1.0)
    else:
        x_adv1 = torch.clamp(x_adv1, 0.0, 1.0)
        x_adv2 = torch.clamp(x_adv2, 0.0, 1.0)
    model1.train()
    model2.train()

    x_adv1 = Variable(torch.clamp(x_adv1, 0.0, 1.0), requires_grad=False)
    x_adv2 = Variable(torch.clamp(x_adv2, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits1 = model1(x_natural)

    logits2 = model2(x_natural)

    logits_adv1 = model1(x_adv1)

    logits_adv2 = model2(x_adv2)

    adv_probs1 = F.softmax(logits_adv1, dim=1)

    adv_probs2 = F.softmax(logits_adv2, dim=1)

    tmp1 = torch.argsort(adv_probs1, dim=1)[:, -2:]

    tmp2 = torch.argsort(adv_probs2, dim=1)[:, -2:]

    new_y1 = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    new_y2 = torch.where(tmp2[:, -1] == y, tmp2[:, -2], tmp2[:, -1])

    loss_adv = F.cross_entropy(logits_adv1, y) + F.nll_loss(torch.log(1.0001 - adv_probs1 + 1e-12), new_y1)+F.cross_entropy(logits_adv2, y) + F.nll_loss(torch.log(1.0001 - adv_probs2 + 1e-12), new_y2)

    nat_probs1 = F.softmax(logits1, dim=1)

    nat_probs2 = F.softmax(logits2, dim=1)

    true_probs1 = torch.gather(nat_probs1, 1, (y.unsqueeze(1)).long()).squeeze()

    true_probs2 = torch.gather(nat_probs2, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs1 + 1e-12), nat_probs1), dim=1) * (1.0000001 - true_probs1))+(1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs2 + 1e-12), nat_probs2), dim=1) * (1.0000001 - true_probs2))
    loss_inter = kl_loss_inter(logits_adv1, logits_adv2)
    loss = loss_adv + float(beta) * loss_robust + 2*loss_inter

    return loss
