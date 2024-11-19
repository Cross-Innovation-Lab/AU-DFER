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
