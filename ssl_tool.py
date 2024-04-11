import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import sys

# from losses import SupConLoss


def ssl_train(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, criterion, epoch, num_epochs, use_cuda=True):

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, inputs_x2, inputs_x3, inputs_x4, targets_x, = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, inputs_x2, inputs_x3, inputs_x4, targets_x, = next(labeled_train_iter)

        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4, targets_u = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4, targets_u = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, args.num_class).scatter_(1, targets_x.view(-1,1).long(), 1)

        if use_cuda:
            inputs_x, inputs_x2, inputs_x3, inputs_x4, targets_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), targets_x.cuda()
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u, _  = model(inputs_u)
            outputs_u2, _ = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
            
        # mixup
        all_inputs  = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input  = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])[0], model(mixed_input[1])[0]]
        for input in mixed_input[2:]:
            logits.append(model(input)[0])

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = torch.cat(logits[0:2], dim=0)
        logits_u = torch.cat(logits[2:],  dim=0)

        Lx, Lu, w = criterion(args, logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/args.train_iteration, num_epochs)

        loss = Lx + w * Lu 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'%(args.dataset, args.inject_portion, args.trigger_type, epoch, num_epochs, batch_idx+1, args.train_iteration, Lx.item(), Lu.item()))
        sys.stdout.flush()


# def ssl_train_with_scl(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, criterion, epoch, num_epochs, use_cuda=True, beta=0.025):

#     labeled_train_iter = iter(labeled_trainloader)
#     unlabeled_train_iter = iter(unlabeled_trainloader)

#     model.train()
    
#     for batch_idx in range(args.train_iteration):
#         try:
#             inputs_x, inputs_x2, inputs_x3, inputs_x4, targets_x, = next(labeled_train_iter)
#         except:
#             labeled_train_iter = iter(labeled_trainloader)
#             inputs_x, inputs_x2, inputs_x3, inputs_x4, targets_x, = next(labeled_train_iter)

#         try:
#             inputs_u, inputs_u2, inputs_u3, inputs_u4, targets_u = next(unlabeled_train_iter)
#         except:
#             unlabeled_train_iter = iter(unlabeled_trainloader)
#             inputs_u, inputs_u2, inputs_u3, inputs_u4, targets_u = next(unlabeled_train_iter)

#         batch_size = inputs_x.size(0)

#         # Transform label to one-hot
#         targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1,1).long(), 1)

#         if use_cuda:
#             inputs_x, inputs_x2, inputs_x3, inputs_x4, targets_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), targets_x.cuda()
#             inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()

#         with torch.no_grad():
#             # compute guessed labels of unlabel samples
#             outputs_u, _  = model(inputs_u)
#             outputs_u2, _ = model(inputs_u2)
#             p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
#             pt = p**(1/args.T)
#             targets_u = pt / pt.sum(dim=1, keepdim=True)
#             targets_u = targets_u.detach()
            
#         labels_u_single = torch.argmax(p, dim=1)

#         # mixup
#         all_inputs  = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
#         all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

#         l = np.random.beta(args.alpha, args.alpha)
#         l = max(l, 1-l)
         
#         idx = torch.randperm(all_inputs.size(0))

#         input_a, input_b = all_inputs, all_inputs[idx]
#         target_a, target_b = all_targets, all_targets[idx]

#         mixed_input  = l * input_a + (1 - l) * input_b
#         mixed_target = l * target_a + (1 - l) * target_b

#         # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
#         mixed_input = list(torch.split(mixed_input, batch_size))
#         mixed_input = interleave(mixed_input, batch_size)

#         logits = [model(mixed_input[0])[0], model(mixed_input[1])[0]]
#         for input in mixed_input[2:]:
#             logits.append(model(input)[0])

#         # put interleaved samples back
#         logits = interleave(logits, batch_size)
#         logits_x = torch.cat(logits[0:2], dim=0)
#         logits_u = torch.cat(logits[2:],  dim=0)

#         Lx, Lu, w = criterion(args, logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/args.train_iteration, num_epochs)

#         _, f1 = model(inputs_u3)
#         _, f2 = model(inputs_u4)
#         f1 = F.normalize(f1, dim=1)
#         f2 = F.normalize(f2, dim=1)
#         features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
#         supconlossfuc = SupConLoss()
#         supconlossfuc = supconlossfuc.cuda()
#         supconloss = supconlossfuc(features, labels_u_single)
        
#         loss = Lx + w * Lu + beta * supconloss

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         sys.stdout.write('\r')
#         sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f  SupCon loss: %.2f'%(args.dataset, args.inject_portion, args.trigger_type, epoch, num_epochs, batch_idx+1, args.train_iteration, Lx.item(), Lu.item(), supconloss.item()))
#         sys.stdout.flush()


def linear_rampup(current, rampup_length=16):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, args, outputs_x, targets_x, outputs_u, targets_u, epoch, num_epochs):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)

# class WeightEMA(object):
#     def __init__(self, model, ema_model, alpha=0.999):
#         self.model = model
#         self.ema_model = ema_model
#         self.alpha = alpha
#         self.params = list(model.state_dict().values())
#         self.ema_params = list(ema_model.state_dict().values())
#         self.wd = 0.02 * args.lr

#         for param, ema_param in zip(self.params, self.ema_params):
#             param.data.copy_(ema_param.data)

#     def step(self):
#         one_minus_alpha = 1.0 - self.alpha
#         for param, ema_param in zip(self.params, self.ema_params):
#             if ema_param.dtype==torch.float32:
#                 ema_param.mul_(self.alpha)
#                 ema_param.add_(param * one_minus_alpha)
#                 # customized weight decay
#                 param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

