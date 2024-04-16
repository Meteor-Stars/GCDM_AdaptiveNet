from Utils.args import args

import torch.optim

torch.manual_seed(args.seed)

import math
from Utils.args import arg_parser

args = arg_parser.parse_args()
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from Utils.dataloader import get_dataloaders
import models
from Utils.loss_functions import _jensen_shannon_reg
from Utils.CDM_module import Uncertainty_aware_Fusion
from Utils.utils import *
set_random_seed(args.seed)
def main():

    global args
    # args.cuda_ = 'cuda:4'
    args.data_root = r'./data/'
    # args.use_valid = True
    args.data = 'MiNi_ImageNet'
    args.data = 'ImageNet'
    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']
    if args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'cifar100' or args.data == 'MiNi_ImageNet':
        args.num_classes = 100
    else:
        args.num_classes = 1000
    # args.growthRate=16
    # args.nChannels=64
    # args.arch='RANet'
    # args.Eval_=True
    # args.step=7
    # args.stepmode='even'
    # args.scale_list='1-2-3-4'
    # args.grFactor='4-2-1-1' #16 16 32 64
    # args.bnFactor='4-2-1-1' #64 64 128 256
    # args.CDM=True
    # args.batch_size= 64
    # args.T=1
    # args.lamda=0.1
    ##8个分类器
    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.scale_list = list(map(int, args.scale_list.split('-')))
    args.nScales = len(args.grFactor)
    print(args)
    args.mode_='anytime_CDM'
    fname = f'{args.data}_{args.arch}_BS{args.batch_size}_{args.mode_}_{args.cuda_}_LR{args.lr}'
    logger = Logger(fname, ask=not args.resume,dir_name=args.save_dir)
    log_ = logger.log
    args.log_=log_
    args.save = logger.logdir
    log_(args)
    device=args.cuda_
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.data.startswith('cifar'):
        IM_SIZE = 32
    else:
        IM_SIZE = 224
    
    args.log_(args.arch)
    model = getattr(models, args.arch)(args)
    args.num_exits = len(model.classifier)

    del(model)
    
    args.log_(args)
    with open('{}/args.txt'.format(args.save), 'w') as f:
        print(args, file=f)

    model = getattr(models, args.arch)(args)
    # model = torch.nn.DataParallel(model,device_ids=[2]).to(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(args.cuda_)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    best_prec1, best_epoch = 0.0, 0
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_prec1=checkpoint['best_prec1']
            best_epoch=checkpoint['best_epoch']

    cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataloaders(args)
    print(len(train_loader), len(val_loader), len(test_loader))
    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1'
              '\tval_prec1\ttrain_prec5\tval_prec5']

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_prec1, train_prec5, lr = train(train_loader, model, criterion, optimizer, epoch)
        val_loss, val_prec1, val_prec5= validate(val_loader, model, criterion)
        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_prec1, val_prec1, train_prec5, val_prec5))
        is_best = val_prec1 > best_prec1

        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            log_('Best var_prec1 {}'.format(best_prec1))

        model_filename = 'last.pth.tar'

        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'best_epoch':best_epoch,
        }, args, is_best, model_filename, scores)

        log_('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))
    return


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.num_exits):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()

    running_lr = None
    for i, (input, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)

        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)
        input = input.to(args.cuda_)
        target = target.cuda(device=args.cuda_)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        if not isinstance(output, list):
            output = [output]

        loss = criterion(output[-1], target_var)
        for j in range(len(output)-1):
            loss += (1-args.lamda)*criterion(output[j], target_var)

        loss_con=0.0
        for i_con in range(len(output)-1):
            if args.train_trick:
                reg_temp=(_jensen_shannon_reg(output[i_con],output[-1],args.trick_T1)*args.lamda+_jensen_shannon_reg(output[i_con],output[-1],args.trick_T2)*args.lamda)/2
            else:
                reg_temp=_jensen_shannon_reg(output[i_con],output[-1],args.T)*args.lamda
            loss_con+=reg_temp
        losses.update(loss.item(), input.size(0))
        loss +=loss_con
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            args.log_('Train Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Loss CON {loss_con}\t''LR {lr_}\t'.format(
                epoch, i + 1, len(train_loader),
                batch_time=batch_time, data_time=data_time,
                loss=losses, loss_con=loss_con, lr_=lr))
    args.log_('[TRAIN DONE] [Time %.3f] [Data %.3f] [LossC %f] [LR %f]' %
         (batch_time.avg, data_time.avg,
          losses.avg,running_lr))

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    top1_nofusion, top5_nofusion = [], []
    for i in range(args.num_exits):
        top1.append(AverageMeter())
        top5.append(AverageMeter())
    for i in range(args.num_exits):
        top1_nofusion.append(AverageMeter())
        top5_nofusion.append(AverageMeter())
    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=args.cuda_)
            input = input.cuda(device=args.cuda_)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]
            view_a_dict = {}
            fusion_e_dict = {}
            fusion_a_dict = {}
            for j in range(len(output)):
                view_a_dict[j] = output[j] + 1
                if j == 0 or j in args.sel_class:
                    fusion_e_dict[j]=output[j]
                else:
                    view_a_temp = {}
                    view_e_temp = []
                    for i in range(j + 1):
                        view_a_temp[i] = view_a_dict[i]
                        view_e_temp.append((view_a_dict[i] - 1).unsqueeze(0))
                    fusion_a_dict[j - 1] = Uncertainty_aware_Fusion(view_a_temp, args.num_classes,
                                                                    balance_term=args.balance_term)
                    fusion_e_dict[j] = fusion_a_dict[j - 1] - 1
            for j in fusion_e_dict.keys():
                prec1, prec5 = accuracy(fusion_e_dict[j].data, target, topk=(1, 5))
                # prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))
            #no fusion
            for j in range(len(output)):
                # prec1, prec5 = accuracy(fusion_e_dict[j].data, target, topk=(1, 5))
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1_nofusion[j].update(prec1.item(), input.size(0))
                top5_nofusion[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    args.log_('[Valid DONE] [Time %.3f] [Data %.3f] [LossC %f]' %
         (batch_time.avg, data_time.avg,
          losses.avg))
    # if args.evalmode is not None:
    for j in range(args.num_exits):
        args.log_(' *FUSION prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))

    for j in range(args.num_exits):
        args.log_(' *NO FUSION prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1_nofusion[j], top5=top5_nofusion[j]))
    return losses.avg, top1_nofusion[-1].avg, top5_nofusion[-1].avg


def save_checkpoint(state, args, is_best, filename, result):
    result_filename = os.path.join(args.save, 'scores.tsv')

    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')


    os.makedirs(args.save, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    args.log_("=> saving checkpoint '{}'".format(model_filename))

    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)

    args.log_("=> saved checkpoint '{}'".format(model_filename))
    return


def load_checkpoint(args):
    model_dir = os.path.join(args.save_pre, 'save_models')

    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()
    else:
        return None

    args.log_("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    args.log_("=> loaded checkpoint '{}'".format(model_filename))
    return state

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='multistep'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data.startswith('cifar'):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()
