import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.ST_Former import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from dataloader.dataset_DFEW import train_data_loader, test_data_loader
from thop import profile

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--data_set', type=int, default=1)
parser.add_argument('--AU_ratio',default=0.25,type=float)
parser.add_argument('--posw_option',default='global', type = str)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
log_txt_path = './log/' + time_str + 'set' + str(args.data_set) + '-log.txt'
log_curve_path = './log/' + time_str + 'set' + str(args.data_set) + '-log.png'
checkpoint_path = './checkpoint/' + time_str + 'set' + str(args.data_set) + '-model.pth'
best_checkpoint_path = './checkpoint/' + time_str + 'set' + str(args.data_set) + '-model_best.pth'


def main():
    best_acc = 0
    best_UAR,best_WAR = 0,0
    recorder = RecorderMeter(args.epochs)
    print('The training time: ' + now.strftime("%m-%d %H:%M"))
    print('The training set: set ' + str(args.data_set))
    with open(log_txt_path, 'a') as f:
        f.write('The training set: set ' + str(args.data_set) + '\n')

    # create model and load pre_trained parameters
    model = GenerateModel()
    model = torch.nn.DataParallel(model).cuda()
    device = torch.device("cuda:0")
    model.to(device)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    train_data = train_data_loader(data_set=args.data_set)
    test_data = test_data_loader(data_set=args.data_set)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        print(inf)
        print('Current learning rate: ', current_learning_rate)

        # train for one epoch
        train_WAR, train_los, train_UAR, train_cm = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_WAR, val_los,val_UAR, val_cm = validate(val_loader, model, criterion, args)

        scheduler.step()

        # remember best acc and save checkpoint
        is_best = (val_WAR + val_UAR) > (best_WAR + best_UAR)
        if is_best:
            best_WAR = val_WAR
            best_UAR = val_UAR
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_WAR': best_WAR,
                         'best_UAR': best_UAR,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best)

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_WAR, val_los, val_WAR)
        recorder.plot_curve(log_curve_path)

        print('The best WAR: {:.3f}'.format(best_WAR))
        print('The best UAR: {:.3f}'.format(best_UAR))
        print('An epoch time: {:.1f}s'.format(epoch_time))
        msg = get_acc_msg(epoch, [train_WAR,train_UAR], train_los, [val_WAR,val_UAR], val_los ,best_WAR, best_UAR, epoch_time)
        with open(log_txt_path, 'a') as f:
            f.write(msg)
        if is_best:
            # print confusion matrix
            cm_msg = get_confusion_msg(val_cm)
            with open(log_txt_path, 'a') as f:
                f.write(cm_msg)
            print(cm_msg)
        #with open(log_txt_path, 'a') as f:
            #f.write('The best WAR: ' + str(best_WAR.item()) + '\n')
            #f.write('The best UAR: ' + str(best_UAR.item()) + '\n')
            #f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    all_targets,all_predictions = [],[]
    for i, (images, target, AU_targets) in enumerate(train_loader):

        images = images.cuda()
        target = target.cuda()
        AU_targets = AU_targets.cuda()
        # compute output
        emo_pred,AU_pred = model(images)
        emo_loss = criterion(emo_pred, target)
        cri_AU = AU_cri_set(target,args.posw_option)
        AU_loss = cri_AU(AU_pred,AU_targets)
        loss = ( 1 - args.AU_ratio ) * emo_loss + args.AU_ratio * AU_loss
        # measure accuracy and record loss
        all_targets.extend(target.cpu().detach().numpy())
        emo_pred_top = torch.argmax(emo_pred, 1).cpu().detach().numpy()
        all_predictions.extend(emo_pred_top)
        acc1, _ = accuracy(emo_pred, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)
    # WAR
    acc1 = accuracy_score(all_targets, all_predictions)
    # UAR
    acc2 = balanced_accuracy_score(all_targets, all_predictions)
    c_m = confusion_matrix(all_targets, all_predictions)
    return acc1, losses.avg,acc2,c_m



def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        all_targets,all_predictions = [],[]
        for i, (images, target, AU_targets) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            AU_targets = AU_targets.cuda()
            emo_pred , AU_pred = model(images)
            #print(emo_pred.size())
            #print(AU_pred.size(),AU_targets.size())
            emo_loss = criterion(emo_pred, target)
            cri_AU = AU_cri_set(target,args.posw_option)
            AU_loss = cri_AU(AU_pred,AU_targets)
            loss = ( 1 - args.AU_ratio ) * emo_loss + args.AU_ratio * AU_loss
            # compute output
            #output = model(images)
            #loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, _ = accuracy(emo_pred, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            all_targets.extend(target.cpu().numpy())
            emo_pred_top = torch.argmax(emo_pred, 1).cpu().numpy()
            all_predictions.extend(emo_pred_top)

    # WAR
    acc1 = accuracy_score(all_targets, all_predictions)
    # UAR
    acc2 = balanced_accuracy_score(all_targets, all_predictions)
    c_m = confusion_matrix(all_targets, all_predictions)
    progress.display(i)
    return acc1, losses.avg,acc2,c_m


def get_acc_msg(epoch, train_acc, train_loss, val_acc, val_loss, best_wa, best_ua, epoch_time):
    msg = """\nEpoch {} Train\t: WA:{:.2%}, \tUA:{:.2%}, \tloss:{:.4f}
                Epoch {} Test\t: WA:{:.2%}, \tUA:{:.2%}, \tloss:{:.4f}
                Epoch {} Best\t: WA:{:.2%}, \tUA:{:.2%}
                Epoch {} Time\t: {:.1f}s\n\n""".format(epoch, train_acc[0], train_acc[1], train_loss, 
                                                        epoch, val_acc[0], val_acc[1], val_loss, 
                                                        epoch, best_wa, best_ua, epoch, epoch_time)
    return msg

def get_confusion_msg(confusion_matrix):
    emotions = ["hap", "sad", "neu", "ang", "sur", "dis", "fea"]
    # change the format of cunfusion matrix to print
    msg = "Confusion Matrix:\n"
    for i in range(len(confusion_matrix)):
        msg += emotions[i]
        for cell in confusion_matrix[i]:
            msg += "\t" + str(cell)
        msg += "\n"
    for emotion in emotions:
        msg += "\t" + emotion
    msg += "\n\n"
    return msg

def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        #print(output.size())
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def vec_mul(a,b):
    res = []
    for i in range(0,len(a)):
        res.append(a[i]*b[i])
    return res

def AU_cri_set(target,posw_option):
    weight = []
    #print(type(weight))
    pos_weight = []
    for emo in target:
        if emo==3:#angry
            weight.append([x*5 for x in [0.16978330014266482, 0.1474484099458089, 0.6385375278419455, 0.15955087911488458, 0.2257386547042889, 0.4972455283651611, 0.15269462267081732, 0.470889570126452, 0.1713443932394683, 0.28682403662377504, 0.17023111167232707, 0.28941226670462944, 0.14671370568908396, 0.12489530852760639, 0.6019091910383833, 0.5023565940498623, 0.2908397350173472, 0.1480589426871144]])
        elif emo==5:#disgust
            weight.append([x*5 for x in [0.2556613250379876, 0.15655681634966592, 0.7784678175136164, 0.13563151267233145, 0.353602732625092, 0.8594560438530944, 0.3484770561236273, 0.7265667054811852, 0.2573762546624107, 0.4833627557878272, 0.26077234063862126, 0.3163550480772833, 0.1807057096038953, 0.1488894096862477, 0.6161944289089377, 0.44359769238463886, 0.2908397350173472, 0.21474224698158992]])
        elif emo==4:#surprise
            weight.append([x*5 for x in [0.3340537870187493, 0.31316038897162674, 0.4508609549567622, 0.2215642997098194, 0.18256790904316805, 0.36794740171201545, 0.14342865607335295, 0.3325723523253354, 0.20291790550542987, 0.24479399356303455, 0.15335881131831128, 0.24479399356303455, 0.14621845929676547, 0.12047847974103684, 0.4921072671708563, 0.5073560806954985, 0.2908397350173472, 0.15234747176484023]])
        elif emo==0:#happy
            weight.append([x*5 for x in [0.26197021302251783, 0.2637489312360733, 0.2964310301204501, 0.13190483696240127, 0.6939537051210115, 0.8330800031639156, 0.17347610820526463, 0.8420321208563623, 0.8882934142237333, 0.7356394920285989, 0.15360866990050534, 0.24664740708336055, 0.17509971698083543, 0.1276861973423668, 0.8057327549388998, 0.41178926421868306, 0.2908397350173472, 0.19836039970638378]])
        elif emo==1:#sad
            weight.append([x*5 for x in [0.2470265618083506, 0.1433979246927842, 0.6881822893577378, 0.11893982182697348, 0.31161634476423544, 0.5253221085983302, 0.12597712051107948, 0.4422623779963743, 0.27645211415344356, 0.42395131504088984, 0.17558170449960314, 0.27142133890425624, 0.1741388696479932, 0.1227455638528141, 0.4485394778155102, 0.41664204112281555, 0.2908397350173472, 0.2121307850723057]])
        elif emo==6:#fear
            weight.append([x*5 for x in [0.3416233199942143, 0.22096625856984287, 0.591078697660434, 0.2210340665261555, 0.19066749128518232, 0.35671400783212986, 0.1239196009045801, 0.3647782154323708, 0.2014525981038694, 0.28661445976904537, 0.21138892966648018, 0.30233592318202696, 0.18447197078258057, 0.12964812274717674, 0.5106650931015151, 0.5397724065827655, 0.2908397350173472, 0.1432476989807878]])
        elif emo==2:#neutral
            weight.append([x*5 for x in [0.19974525492944142, 0.17419319637385966, 0.4053456334801515, 0.13706230525786608, 0.16464150662864904, 0.3564013123982043, 0.1117065857762667, 0.30225692349830036, 0.17199394819786643, 0.28648655803768563, 0.14721999753281687, 0.22595757296022984, 0.14108230584951664, 0.11916179838996314, 0.31830472814177574, 0.36213049142283893, 0.2908397350173472, 0.16214811486455724]])
    weight = torch.Tensor(weight)
    if posw_option == 'global':
        pos_weight = np.array([8.123028391167193, 10.40521645603657, 1.0299431287937402, 1.904973346878674, 4.991242525542634, 2.9253478113335594, 31.506833567505428, 2.493520755545794, 7.428694442604491, 2.5988969808385773, 4.979137299126022, 2.8861470803811384, 12.160409556313994, 6.5054854311666865, 5.102436217149434, 9.670691823899372, 101.28938906752411, 8.483380533611566])
    if posw_option == 'distinct':
        for emo in target:
            if emo==3:#angry
                pos_weight.append(np.array([4.928812812224508, 4.114906832298137, 0.9234234234234234, 1.4543961558346765, 4.781224255883091, 2.435997871208089, 12.75189571440743, 1.44547134935305, 13.252185430463577, 3.1313061506565307, 3.579413266753674, 2.578767654819184, 5.92967542503864, 5.292631578947368, 2.6113572291582763, 4.433813627794237, 55.85311729482212, 4.549840112780662]))
            elif emo==5:#disgust
                pos_weight.append(np.array([4.229214780600462, 4.09392575928009, 0.7474435655026047, 2.1834797891036906, 5.054144385026738, 1.6915304606240713, 10.307116104868914, 2.2381122631390777, 10.731865284974093, 3.210599721059972, 4.137266023823029, 3.012848914488259, 5.983037779491133, 6.148382004735596, 3.6374807987711213, 5.387165021156559, 27.391849529780565, 3.2964895635673623]))
            elif emo==4:#surprise
                pos_weight.append(np.array([7.116157728166966, 6.34866790582404, 0.9600900658968373, 0.761682850299846, 17.648977987421382, 5.163029358274876, 21.278938718008924, 6.183435536376713, 35.41059094397544, 6.815336463223788, 6.358355951919348, 4.08091030789826, 9.571078431372548, 7.167857450288371, 5.090243902439024, 8.104394549990404, 94.26706827309236, 6.122504128509233]))
            elif emo==0:#happy
                pos_weight.append(np.array([4.648862512363996, 4.162923411588553, 1.83239825175909, 2.5131103421760863, 1.0496164371270917, 1.191954326288771, 15.430099793221252, 0.636697444899202, 1.0033104960263086, 0.7217365088935785, 3.69328950409615, 2.862698681095705, 6.9568094740508535, 4.616744014506562, 2.920370688175734, 5.462006293978289, 57.59313882654697, 4.170958066889254]))
            elif emo==1:#sad
                pos_weight.append(np.array([5.6823671940967, 4.992658194508461, 0.5149984185907146, 2.6989371862870613, 3.1921004145555045, 1.6714365386873313, 19.07732186190161, 2.1056834274599465, 7.311081685767773, 1.9735191296108734, 5.039584577609662, 3.51384996900186, 8.076543333000897, 6.52494935714581, 3.628397792864953, 5.6926866933852995, 70.26898981989036, 5.224216933388045]))
            elif emo==6:#fear
                pos_weight.append(np.array([5.96072648535643, 5.154494891980657, 0.7569813845450734, 1.5509251975236857, 7.060588375159415, 2.779909786630176, 17.84796563052818, 2.8554517069539505, 15.333362533397574, 3.2621352565348083, 4.923954312221004, 3.5370230679384855, 7.644713355124371, 5.762544656620061, 3.9785987023043443, 6.670773851153989, 57.87385538364383, 5.229860670252932]))
            elif emo==2:#neutral
                pos_weight.append(np.array([6.4774986002239645, 5.6010812480692, 0.8710279064472912, 1.9167337801498792, 7.8117860530331145, 2.7351547887496284, 21.302160526041124, 3.19245786489297, 20.999073406774425, 2.574808023689626, 5.714757086292502, 3.906024704963953, 8.191199242945629, 5.206488904380156, 4.249791165053314, 7.363091976516634, 81.4053220208253, 4.963300960035722]))
    if posw_option == 'minor':
        for emo in target:
            if emo==3:#angry
                pos_weight.append(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
            elif emo==5:#disgust
                pos_weight.append(np.array([4.229214780600462, 4.09392575928009, 0.7474435655026047, 2.1834797891036906, 5.054144385026738, 1.6915304606240713, 10.307116104868914, 2.2381122631390777, 10.731865284974093, 3.210599721059972, 4.137266023823029, 3.012848914488259, 5.983037779491133, 6.148382004735596, 3.6374807987711213, 5.387165021156559, 27.391849529780565, 3.2964895635673623]))
            elif emo==4:#surprise
                pos_weight.append(np.array([7.116157728166966, 6.34866790582404, 0.9600900658968373, 0.761682850299846, 17.648977987421382, 5.163029358274876, 21.278938718008924, 6.183435536376713, 35.41059094397544, 6.815336463223788, 6.358355951919348, 4.08091030789826, 9.571078431372548, 7.167857450288371, 5.090243902439024, 8.104394549990404, 94.26706827309236, 6.122504128509233]))
            elif emo==0:#happy
                pos_weight.append(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
            elif emo==1:#sad
                pos_weight.append(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
            elif emo==6:#fear
                pos_weight.append(np.array([6.047279943964511, 5.658614604015001, 0.9386300138090498, 1.6440804169769174, 10.258299142111152, 2.639197009886665, 17.34305682163476, 4.947487684729064, 15.796605453533667, 3.207639227713111, 6.6501077176530226, 3.6865150221256116, 12.636096679466908, 7.38081354990976, 4.699301359516617, 6.53645443196005, 152.21573604060913, 5.966762839007502]))
            elif emo==2:#neutral
                pos_weight.append(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
    pos_weight = np.array(pos_weight)
    pos_weight=torch.from_numpy(pos_weight)
    criterion_AU = torch.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
    if torch.cuda.is_available():
        criterion_AU = criterion_AU.cuda()
        cudnn.benchmark = True
        return criterion_AU

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


if __name__ == '__main__':
    main()
