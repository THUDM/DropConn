import torch
import torch.nn.functional as F

import pdb


def consis_loss(logps, temp=0.5, lam=1.0, conf=0.):
    input_list = []
    for p in logps:
        p = p.view(-1, 1)
        label1_pro = torch.sigmoid(p)
        label0_pro = 1 - label1_pro
        label_pro = torch.cat([label0_pro, label1_pro], dim=1)
        input_list.append(label_pro)
    ps = input_list
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)
    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=-1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1)[avg_p.max(1)[0] > conf])
    loss = loss / len(ps)
    return loss


def flag_cr(epoch, num_batch, model_forward, perturb_shape, y, args, optimizer, device, criterion):
    model, train_forward, test_forward = model_forward
    model.train()
    optimizer.zero_grad()
    alpha, lam = 1, 1
    train_perturb_shape, test_perturb_shape = perturb_shape
    train_perturb = torch.FloatTensor(*train_perturb_shape).uniform_(-args.step_size, args.step_size).to(device)
    test_perturb = torch.FloatTensor(*test_perturb_shape).uniform_(-args.step_size, args.step_size).to(device)
    train_perturb.requires_grad_()
    test_perturb.requires_grad_()

    train_pred = train_forward(train_perturb)
    loss_train = criterion(train_pred, y)
    test_pred_list = []
    if args.dataset in ['ogbg-moltox21']:
        conf = 0.
    elif args.dataset in ['ogbg-moltoxcast']:
        conf = 0.5
    else:
        conf = 0.95
    for k in range(args.sample):
        test_pred = test_forward(test_perturb)
        test_pred_list.append(test_pred)
    loss_consis = consis_loss(test_pred_list, conf=args.conf)
    lam = min(lam, (lam * float(num_batch) / args.warmup))
    loss = alpha * loss_train + lam * loss_consis
    loss /= args.step_m

    for _ in range(args.step_m - 1):
        loss.backward()
        train_perturb_data = train_perturb.detach() + args.step_size * torch.sign(train_perturb.grad.detach())
        train_perturb.data = train_perturb_data.data
        train_perturb.grad[:] = 0
        train_pred = train_forward(train_perturb)
        loss_train = criterion(train_pred, y)
        loss_train /= args.step_m
        test_pred_list = []
        test_perturb_data = test_perturb.detach() + args.step_size * torch.sign(test_perturb.grad.detach())
        test_perturb.data = test_perturb_data.data
        test_perturb.grad[:] = 0
        for k in range(args.sample):
            test_pred = test_forward(test_perturb)
            test_pred_list.append(test_pred)
        loss_consis = consis_loss(test_pred_list, conf=args.conf)
        lam = min(lam, (lam * float(num_batch) / args.warmup))
        # lam = min(lam, epoch * 0.1)
        loss = alpha * loss_train + lam * loss_consis
        loss /= args.step_m
    loss.backward()
    optimizer.step()

    return loss, train_pred


def flag_biased(model_forward, perturb_shape, y, args, optimizer, device, criterion, training_idx):
    unlabel_idx = list(set(range(perturb_shape[0])) - set(training_idx))

    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb.data[unlabel_idx] *= args.amp
    perturb.requires_grad_()
    out = forward(perturb)
    loss = criterion(out, y)
    loss /= args.m

    for _ in range(args.m - 1):
        loss.backward()

        perturb_data_training = perturb[training_idx].detach() + args.step_size * torch.sign(
            perturb.grad[training_idx].detach())
        perturb.data[training_idx] = perturb_data_training.data

        perturb_data_unlabel = perturb[unlabel_idx].detach() + args.amp * args.step_size * torch.sign(
            perturb.grad[unlabel_idx].detach())
        perturb.data[unlabel_idx] = perturb_data_unlabel.data

        perturb.grad[:] = 0
        out = forward(perturb)
        loss = criterion(out, y)
        loss /= args.m

    loss.backward()
    optimizer.step()

    return loss, out


def flag(model_forward, perturb_shape, y, args, optimizer, device, criterion):
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb.requires_grad_()
    out = forward(perturb)
    loss = criterion(out, y)
    loss /= args.m

    for _ in range(args.m - 1):
        loss.backward()
        perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb)
        loss = criterion(out, y)
        loss /= args.m

    loss.backward()
    optimizer.step()

    return loss, out


def flag_products(model, clean, y, adjs, args, optimizer, device, criterion, train_idx=None):
    model.train()
    if train_idx is not None:
        model_forward = lambda x: model(x, adjs)[train_idx]
    else:
        model_forward = lambda x: model(x, adjs)
    optimizer.zero_grad()

    perturb_t = torch.FloatTensor(*clean[:args.batch_size].shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb_un = torch.FloatTensor(*clean[args.batch_size:].shape).uniform_(-args.amp * args.step_size,
                                                                            args.amp * args.step_size).to(device)
    perturb_t.requires_grad_()
    perturb_un.requires_grad_()

    perturb = torch.cat((perturb_t, perturb_un), dim=0)
    out = model_forward(clean + perturb)
    loss = criterion(out, y)
    loss /= args.m

    for _ in range(args.m - 1):
        loss.backward()

        perturb_data_t = perturb_t.detach() + args.step_size * torch.sign(perturb_t.grad.detach())
        perturb_t.data = perturb_data_t.data
        perturb_t.grad[:] = 0

        perturb_data_un = perturb_un.detach() + args.amp * args.step_size * torch.sign(perturb_un.grad.detach())
        perturb_un.data = perturb_data_un.data
        perturb_un.grad[:] = 0

        perturb = torch.cat((perturb_t, perturb_un), dim=0)
        out = model_forward(clean + perturb)
        loss = criterion(out, y)
        loss /= args.m

    loss.backward()
    optimizer.step()

    return loss, out
