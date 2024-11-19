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
from dataloader.dataset_FERV39K import train_data_loader, test_data_loader
from thop import profile

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
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

args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
log_txt_path = './log/' + time_str + 'log.txt'
log_curve_path = './log/' + time_str + 'log.png'
checkpoint_path = './checkpoint/' + time_str + 'model.pth'
best_checkpoint_path = './checkpoint/' + time_str + 'model_best.pth'

def main():
    best_acc = 0
    best_UAR,best_WAR = 0,0
    recorder = RecorderMeter(args.epochs)
    #print('The training time: ' + now.strftime("%m-%d %H:%M"))
    #print('The training set: set ' + str(args.data_set))
    #with open(log_txt_path, 'a') as f:
        #f.write('The training set: set ' + str(args.data_set) + '\n')

    # create model and load pre_trained parameters
    model = GenerateModel()
    model = torch.nn.DataParallel(model).cuda()
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
        is_best = (val_WAR + val_UAR) > (best_WAR + best_UAR)
        if is_best:
            best_WAR = val_WAR
            best_UAR = val_UAR
        scheduler.step()

        # remember best acc and save checkpoint
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
        
        '''
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')
        '''


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
        #print(AU_targets)
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
            #AU_targets = np.array(AU_targets)
            #AU_targets = torch.from_numpy(AU_targets).t()
            AU_targets = AU_targets.cuda()
            # compute output
            emo_pred,AU_pred = model(images)
            emo_loss = criterion(emo_pred, target)
            cri_AU = AU_cri_set(target,args.posw_option)
            AU_loss = cri_AU(AU_pred,AU_targets)
            loss = ( 1 - args.AU_ratio ) * emo_loss + args.AU_ratio * AU_loss

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
def AU_cri_set(target,posw_option):
    weight = []
    pos_weight = []
    for emo in target:
        if emo==3:#angry
            weight.append([x*5 for x in [0.3101259978601843, 0.2666147102288715, 0.13845042098707322, 0.29109966182655833, 0.3857435256997118, 0.28414777060394286, 0.27748466368999686, 0.3112494485722471, 0.31289378029762777, 0.41722449809305917, 0.31092253027646627, 0.4175421849893703, 0.26506202105256416, 0.21584591344096649, 0.1741026302426755, 0.27878341068306967, 1.74694374224107,1.74694374224107]])
        elif emo==5:#disgust
            weight.append([x*5 for x in[0.40721574689736123, 0.28523795317106965, 0.03657068162975569, 0.24077926879096198, 0.4043433665405009, 0.008576630671049652, 0.406733027316152, 0.06688077218181326, 0.4080793452037397, 0.29855891816981045, 0.4096776471203334, 0.4166519181063416, 0.3286751116548074, 0.2696385610908488, 0.15988227254676246, 0.3378189743033448, 1.74694374224107,1.74694374224107]])
        elif emo==4:#surprise
            weight.append([x*5 for x in[0.4122681573130717, 0.4171378321149753, 0.33093810219201664, 0.3817419876877954, 0.33165367004699836, 0.3967823898720178, 0.258170231067692, 0.41274049513641964, 0.36069688657829474, 0.400841402022718, 0.2790962209229562, 0.400841402022718, 0.26396301036737124, 0.20524107004816294, 0.28951133665255213, 0.27351215699444636, 1.74694374224107,1.74694374224107]])
        elif emo==0:#happy
            weight.append([x*5 for x in[0.4102062600389829, 0.41095770811632504, 0.41803458042778374, 0.2322705966248465, 0.09066378229764688, 0.015180147816263303, 0.31660281950777386, 0.012679725458816142, 0.0038333838772364385, 0.06089417471969754, 0.2793389235402054, 0.40204037116646585, 0.3193842331584253, 0.2224482236651945, 0.024549694471830814, 0.3658364744360839, 1.74694374224107,1.74694374224107]])
        elif emo==1:#sad
            weight.append([x*5 for x in[0.40229523588843863, 0.257963948348277, 0.09524529767453874, 0.20150506655830994, 0.41733760982735046, 0.25443398251236693, 0.21844718328633875, 0.33906684205510706, 0.41518877136319954, 0.3555757319693506, 0.3202021146171915, 0.41374786323570883, 0.3177430994906438, 0.21070730310497668, 0.33315401347369766, 0.3618163084198552, 1.74694374224107,1.74694374224107]])
        elif emo==6:#fear
            weight.append([x*5 for x in[0.4095512663816923, 0.3812487634270026, 0.18511101108970013, 0.38121540370466694, 0.34398230195429214, 0.40285188777671566, 0.21351919840164818, 0.39858175147539276, 0.3589580490082006, 0.4171955067776652, 0.3709072830104787, 0.41803804256344024, 0.3346435315500807, 0.22704054056925538, 0.2700121671610338, 0.23902276499259092, 1.74694374224107,1.74694374224107]])
        elif emo==2:#neutral
            weight.append([x*5 for x in[0.3565903724341121, 0.3178470861967956, 0.37101264160590863, 0.24400201093767518, 0.30076189712205165, 0.4030082784253525, 0.1836916286924921, 0.41804519923131944, 0.31402834992942114, 0.41717757201890737, 0.2661327833834797, 0.3859463577046634, 0.2529205867172519, 0.20204530297384235, 0.41630794601879584, 0.3999640767999124, 1.74694374224107,1.74694374224107]])
    weight = torch.Tensor(weight)
    if posw_option == 'global':
        pos_weight = np.array([4.789480561907873, 4.554315802670344, 1.4431454725929194, 1.2930490140261892, 10.91287980639957, 2.960842161727236, 12.93426639408712, 4.830240821160679, 9.943312337902928, 3.558962749536942, 5.529216712106698, 3.1841620626151017, 10.882526485181709, 8.180273518441775, 3.816699282452707, 5.920877919237679, 128.54385964912282, 5.021610601427115])
    if posw_option == 'distinct':
        for emo in target:
            if emo==3:#angry
                pos_weight.append(np.array([5.302377957822691, 4.934069565696346, 0.9931917412861688, 1.7729132885491468, 6.890146105679446, 2.4329960965506254, 14.58004338394794, 3.350684488955296, 14.718704406186168, 3.104384928949674, 5.258444915623457, 3.049844939385396, 9.561835204156658, 6.84933153619176, 3.5544705136334813, 5.460348394447276, 118.57380688124306, 5.117713863888022]))
            elif emo==5:#disgust
                pos_weight.append(np.array([4.931708599982467, 4.588288734720845, 0.9527561327561328, 2.4336242768699887, 4.1825214460784315, 1.9008788853161844, 13.448644031603672, 2.618535750574897, 8.57043847241867, 2.3589654487688643, 5.042957935161204, 3.0051497573102877, 9.752105514063246, 5.854031604538087, 3.6751191874524975, 5.442254593925545, 109.20032573289902, 4.523510204081632]))
            elif emo==4:#surprise
                pos_weight.append(np.array([4.789480561907873, 4.554315802670344, 1.4431454725929194, 1.2930490140261892, 10.91287980639957, 2.960842161727236, 12.934266394087121, 4.830240821160679, 9.943312337902928, 3.5589627495369416, 5.529216712106698, 3.1841620626151013, 10.882526485181709, 8.180273518441775, 3.8166992824527073, 5.920877919237679, 128.5438596491228, 5.021610601427115]))
            elif emo==0:#happy
                pos_weight.append(np.array([4.737603191312538, 4.541120445181658, 1.9399841017488075, 3.0183148051141764, 1.8016286336521163, 1.0785805325010287, 12.648500823723229, 1.2691427006299645, 1.6937363437727604, 1.317200331162874, 4.711181580035848, 2.9899824693213124, 10.53271340275071, 5.179984484096199, 3.398861609039164, 5.603832541529828, 123.02155688622754, 4.838200473559589]))
            elif emo==1:#sad
                pos_weight.append(np.array([6.529455713698681, 6.403070594284875, 0.9404640415904496, 2.296684603935295, 6.959404493938317, 2.2288739727344087, 17.447373238147538, 4.3383833033160295, 11.864181771764105, 2.7964965153512904, 7.251361198673599, 3.9084577356744514, 13.404059172443365, 8.52803252339983, 4.647724725397893, 6.739948542682693, 163.66993464052288, 6.3258459637262385]))
            elif emo==6:#fear
                pos_weight.append(np.array([6.047279943964511, 5.658614604015001, 0.9386300138090498, 1.6440804169769174, 10.258299142111152, 2.639197009886665, 17.34305682163476, 4.947487684729064, 15.796605453533667, 3.207639227713111, 6.6501077176530226, 3.6865150221256116, 12.636096679466908, 7.38081354990976, 4.699301359516617, 6.53645443196005, 152.21573604060913, 5.966762839007502]))
            elif emo==2:#neutral
                pos_weight.append(np.array([5.821125337847061, 5.533247914651048, 1.3883817189396908, 2.207288738270369, 7.343384759233286, 2.429757851965737, 15.828382838283828, 4.222092172640819, 10.11337959256294, 2.900508929686524, 6.057539122083498, 3.320439218398755, 11.676357179096906, 6.140436112143123, 4.243682840832774, 6.373498598199793, 133.32849462365593, 5.4467695324594905]))
    if posw_option == 'minor':
        for emo in target:
            if emo==3:#angry
                pos_weight.append(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
            elif emo==5:#disgust
                pos_weight.append(np.array([4.931708599982467, 4.588288734720845, 0.9527561327561328, 2.4336242768699887, 4.1825214460784315, 1.9008788853161844, 13.448644031603672, 2.618535750574897, 8.57043847241867, 2.3589654487688643, 5.042957935161204, 3.0051497573102877, 9.752105514063246, 5.854031604538087, 3.6751191874524975, 5.442254593925545, 109.20032573289902, 4.523510204081632]))
            elif emo==4:#surprise
                pos_weight.append(np.array([4.789480561907873, 4.554315802670344, 1.4431454725929194, 1.2930490140261892, 10.91287980639957, 2.960842161727236, 12.934266394087121, 4.830240821160679, 9.943312337902928, 3.5589627495369416, 5.529216712106698, 3.1841620626151013, 10.882526485181709, 8.180273518441775, 3.8166992824527073, 5.920877919237679, 128.5438596491228, 5.021610601427115]))
            elif emo==0:#happy
                pos_weight.append(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
            elif emo==1:#sad
                pos_weight.append(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
            elif emo==6:#fear
                pos_weight.append(np.array([5.821125337847061, 5.533247914651048, 1.3883817189396908, 2.207288738270369, 7.343384759233286, 2.429757851965737, 15.828382838283828, 4.222092172640819, 10.11337959256294, 2.900508929686524, 6.057539122083498, 3.320439218398755, 11.676357179096906, 6.140436112143123, 4.243682840832774, 6.373498598199793, 133.32849462365593, 5.4467695324594905]))
            elif emo==2:#neutral
                pos_weight.append(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
    pos_weight = np.array(pos_weight)
    pos_weight = torch.from_numpy(pos_weight)
    criterion_AU = torch.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
    if torch.cuda.is_available():
        criterion_AU = criterion_AU.cuda()
        cudnn.benchmark = True
        return criterion_AU

if __name__ == '__main__':
    main()
