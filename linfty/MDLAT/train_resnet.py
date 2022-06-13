import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import itertools

import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import *
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr-schedule', default='multistep', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0.0001, type=float)
    parser.add_argument('--lr-max', default=0.05, type=float)
    parser.add_argument('--weight-decay', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--alpha', default=2, type=int, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='train_resnet4', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()

def kl_loss(output1, output2):
    criterion_kl = nn.KLDivLoss(size_average=False)
    loss = (1.0 / output1.size(0))*criterion_kl(F.log_softmax(output1, dim=1), F.softmax(output2, dim=1)) + (1.0 / output1.size(0))*criterion_kl(F.log_softmax(output2, dim=1), F.softmax(output1, dim=1))
    return loss    

def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    model1 = ResNet18().cuda()
    model1.train()

    model2 = ResNet18().cuda()
    model2.train()


    opt = torch.optim.SGD(itertools.chain(model1.parameters(), model2.parameters()), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps*3 / 4, lr_steps * 9 / 10], gamma=0.1)

    # Training
    start_train_time = time.time()
    index_list = []
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            delta1 = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                for i in range(len(epsilon)):
                    delta1[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                delta1.data = clamp(delta1, lower_limit - X, upper_limit - X)
            delta1.requires_grad = True
            delta2 = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                for i in range(len(epsilon)):
                    delta2[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                delta2.data = clamp(delta2, lower_limit - X, upper_limit - X)
            delta2.requires_grad = True
            for _ in range(args.attack_iters):
                output1 = model1(X + delta1)
                output2 = model2(X + delta2)
                loss = criterion(output1, y)+criterion(output2, y)+kl_loss(output1, output2)
                loss.backward()
                grad1 = delta1.grad.detach()
                delta1.data = clamp(delta1 + alpha * torch.sign(grad1), -epsilon, epsilon)
                delta1.data = clamp(delta1, lower_limit - X, upper_limit - X)
                delta1.grad.zero_()
                grad2 = delta2.grad.detach()
                delta2.data = clamp(delta2 + alpha * torch.sign(grad2), -epsilon, epsilon)
                delta2.data = clamp(delta2, lower_limit - X, upper_limit - X)
                delta2.grad.zero_()
            delta1 = delta1.detach()
            delta2 = delta2.detach()
            output1 = model1(X + delta1)
            output2 = model2(X + delta2)
            loss = 0.5*criterion(output1, y)+0.5*criterion(output2, y)+0.5*kl_loss(output1, output2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output1.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
        train_time = time.time()
        torch.save(model1.state_dict(), os.path.join(args.out_dir, 'model1'+str(epoch)+'.pth'))
        torch.save(model2.state_dict(), os.path.join(args.out_dir, 'model2'+str(epoch)+'.pth'))
        logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

        # Evaluation

        model_test = ResNet18().cuda()
        model_test.load_state_dict(model1.state_dict())
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 20, 1)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)

        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)



if __name__ == "__main__":
    main()
