import torch
import torch.nn as nn
import time
import os
import seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import torch.backends.cudnn as cudnn
from models import *
from datasets import *
from utils import *


class Solver(object):
    def __init__(self, args):
        """Init the global settings including device, seed, models, dataloaders, crterions, optimizers and schedulers

        Args:
            args
        """
        super(Solver, self).__init__()

        self.args = args
        self.log_path = os.path.join(self.args.output_path, "log.txt")
        self.emotions = ["hap", "sad", "neu", "ang", "sur", "dis", "fea"]
        self.best_wa = 0
        self.best_ua = 0

        # init cuda
        if len(self.args.gpu_ids) > 0:
            torch.cuda.set_device(self.args.gpu_ids[0])
        self.device = torch.device(
            'cuda:%d' % self.args.gpu_ids[0] if self.args.gpu_ids else 'cpu')

        # set seed
        seed = self.args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # init model
        self.model = create_model(self.args)
        if len(self.args.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, self.args.gpu_ids)
        self.model.to(self.device)

        # init dataloader
        self.train_dataloader = create_dataloader(self.args, "train")
        print('length of train loader',len(self.train_dataloader))
        self.test_dataloader = create_dataloader(self.args, "test")
        print('length of test loader',len(self.test_dataloader))

        # init criterion
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.args.label_smoothing).to(self.device)

        # init optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.args.lr,
                                           eps=self.args.eps,
                                           weight_decay=self.args.weight_decay)
        self.scheduler = build_scheduler(
            self.args, self.optimizer, len(self.train_dataloader))

        # resume
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
            self.args.start_epoch = checkpoint['epoch'] + 1
            self.best_wa = checkpoint['best_wa']
            self.best_ua = checkpoint['best_ua']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def AU_cri_set(self,target):
        weight, pos_weight = [], []
        #print(type(weight))
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
        if self.args.posw_option == 'global' and self.args.dataset == 'DFEW':
            pos_weight = np.array([8.123028391167193, 10.40521645603657, 1.0299431287937402, 1.904973346878674, 4.991242525542634, 2.9253478113335594, 31.506833567505428, 2.493520755545794, 7.428694442604491, 2.5988969808385773, 4.979137299126022, 2.8861470803811384, 12.160409556313994, 6.5054854311666865, 5.102436217149434, 9.670691823899372, 101.28938906752411, 8.483380533611566])
        if self.args.posw_option == 'global' and self.args.dataset == 'FERV39K':
            pos_weight = np.array([4.789480561907873, 4.554315802670344, 1.4431454725929194, 1.2930490140261892, 10.91287980639957, 2.960842161727236, 12.93426639408712, 4.830240821160679, 9.943312337902928, 3.558962749536942, 5.529216712106698, 3.1841620626151017, 10.882526485181709, 8.180273518441775, 3.816699282452707, 5.920877919237679, 128.54385964912282, 5.021610601427115])
        if self.args.posw_option == 'distinct' and self.args.dataset == 'DFEW':
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
        if self.args.posw_option == 'distinct' and self.args.dataset == 'FERV39K':
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
        if self.args.posw_option == 'minor' and self.args.dataset == 'DFEW':
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
        if self.args.posw_option == 'minor' and self.args.dataset == 'FERV39K':
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
        pos_weight=torch.from_numpy(pos_weight)
        criterion_AU = torch.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
        if torch.cuda.is_available():
            criterion_AU = criterion_AU.cuda()
            cudnn.benchmark = True
        return criterion_AU

    def run(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            inf = '********************' + str(epoch) + '********************'
            start_time = time.time()

            with open(self.log_path, 'a') as f:
                f.write(inf + '\n')
            print(inf)

            # train the model for one epoch
            train_acc, train_loss = self.train(epoch)
            # validate the model
            val_acc, val_loss = self.validate(epoch)

            # remember best acc and save checkpoint
            is_best = val_acc[0]+val_acc[1] >self.best_wa+self.best_ua
            if is_best:
                self.best_wa = val_acc[0]
                self.best_ua = val_acc[1]
            #self.best_wa = max(val_acc[0], self.best_wa)
            #self.best_ua = max(val_acc[1], self.best_ua)
            self.save({'epoch': epoch,
                       'state_dict': self.model.state_dict(),
                       'best_wa': self.best_wa,
                       'best_ua': self.best_ua,
                       'optimizer': self.optimizer.state_dict(),
                       'args': self.args}, is_best)

            # print and save log
            epoch_time = time.time() - start_time
            msg = self.get_acc_msg(epoch, train_acc, train_loss, val_acc, val_loss,
                                   self.best_wa, self.best_ua, epoch_time)
            with open(self.log_path, 'a') as f:
                f.write(msg)
            print(msg)

            if is_best:
                # print confusion matrix
                cm_msg = self.get_confusion_msg(val_acc[2])
                with open(self.log_path, 'a') as f:
                    f.write(cm_msg)
                print(cm_msg)

                # convert confusion matrix to heatmap
                cm = []
                for row in val_acc[2]:
                    row = row / np.sum(row)
                    cm.append(row)
                fig_path = os.path.join(self.args.output_path, "fig_best.png")
                ax = seaborn.heatmap(
                    cm, xticklabels=self.emotions, yticklabels=self.emotions, cmap='rocket_r')
                figure = ax.get_figure()
                # save the heatmap
                figure.savefig(fig_path)
                plt.close()

        return self.best_ua, self.best_ua

    def train(self, epoch):
        """ Train the model for one eopch
        """
        self.model.train()
        all_pred, all_target = [], []
        all_loss = 0

        for i, (images, target, AU_labels) in enumerate(self.train_dataloader):
            #print(target)

            print("Training epoch \t{}: {}\\{}".format(
                epoch, i + 1, len(self.train_dataloader)), end='\r')
            images = images.to(self.device)
            target = target.to(self.device)
            '''
            print(target.size())
            #print(AU_labels.size())
            for i in range(0,len(AU_labels)):
                AU_labels[i] = AU_labels[i].tolist()
                AU_labels[i] = torch.Tensor(AU_labels[i]).t()
                print(AU_labels[i].size())
            print(len(AU_labels))
            #print(AU_labels)
            AU_labels = AU_labels.to(self.device)
            '''
            #AU_labels = torch.Tensor(AU_labels)
            AU_labels = AU_labels.to(self.device)
            output,AU_output = self.model(images)
            #print('size of AU_output',AU_output.size())
            #print("AU_output:",AU_output)
            #print(len(AU_labels),AU_output.size())
            loss = self.criterion(output, target)
            cri_AU = self.AU_cri_set(target)
            loss_AU = cri_AU(AU_output,AU_labels)
            #print(loss,loss_AU)
            loss = (1-self.args.AU_weight)*loss+self.args.AU_weight*loss_AU

            pred = torch.argmax(output, 1).cpu().detach().numpy()
            target = target.cpu().numpy()

            all_pred.extend(pred)
            all_target.extend(target)
            all_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step_update(epoch * len(self.train_dataloader) + i)
        # WAR
        acc1 = accuracy_score(all_target, all_pred)
        # UAR
        acc2 = balanced_accuracy_score(all_target, all_pred)
        loss = all_loss / len(self.train_dataloader)
        #loss = np.array(loss)
        return [acc1, acc2], loss

    def validate(self, epoch):
        """Validate the model for one epoch
        """
        self.model.eval()

        all_pred, all_target = [], []
        all_loss = 0

        for i, (images, target,AU_labels) in enumerate(self.test_dataloader):

            print("Testing epoch \t{}: {}\\{}".format(
                epoch, i + 1, len(self.test_dataloader)), end='\r')

            images = images.to(self.device)
            target = target.to(self.device)
            #print(AU_labels.size())
            #for i in range(0,len(AU_labels)):
                #AU_labels[i] = AU_labels[i].tolist()
            #AU_labels = torch.Tensor(AU_labels).t()
            #AU_labels = torch.Tensor(AU_labels)
            
            #print(AU_labels)
            AU_labels = AU_labels.to(self.device)
            #print(target)

            with torch.no_grad():
                output,AU_output = self.model(images)
                #output = self.model(images)

            loss = self.criterion(output, target)
            cri_AU = self.AU_cri_set(target)
            loss_AU = cri_AU(AU_output,AU_labels)
            

            pred = torch.argmax(output, 1).cpu().detach().numpy()
            target = target.cpu().numpy()

            all_pred.extend(pred)
            all_target.extend(target)
            all_loss += (1-self.args.AU_weight)*loss.item()+self.args.AU_weight*loss_AU.item()
            #all_loss += 0.75*loss.item()+0.25*loss_AU.item()

        # WAR
        acc1 = accuracy_score(all_target, all_pred)

        # UAR
        acc2 = balanced_accuracy_score(all_target, all_pred)

        c_m = confusion_matrix(all_target, all_pred)
        loss = all_loss / len(self.test_dataloader)

        return [acc1, acc2, c_m], loss

    def save(self, state, is_best):
        # save the best model
        if is_best:
            checkpoint_path = os.path.join(
                self.args.output_path, "model_best.pth")
            torch.save(state, checkpoint_path)

        # save the latest model for resume
        checkpoint_path = os.path.join(
            self.args.output_path, "model_latest.pth")
        torch.save(state, checkpoint_path)

    def get_acc_msg(self, epoch, train_acc, train_loss, val_acc, val_loss, best_wa, best_ua, epoch_time):
        msg = """\nEpoch {} Train\t: WA:{:.2%}, \tUA:{:.2%}, \tloss:{:.4f}
                   Epoch {} Test\t: WA:{:.2%}, \tUA:{:.2%}, \tloss:{:.4f}
                   Epoch {} Best\t: WA:{:.2%}, \tUA:{:.2%}
                   Epoch {} Time\t: {:.1f}s\n\n""".format(epoch, train_acc[0], train_acc[1], train_loss, 
                                                          epoch, val_acc[0], val_acc[1], val_loss, 
                                                          epoch, best_wa, best_ua, epoch, epoch_time)
        return msg

    def get_confusion_msg(self, confusion_matrix):
        # change the format of cunfusion matrix to print
        msg = "Confusion Matrix:\n"
        for i in range(len(confusion_matrix)):
            msg += self.emotions[i]
            for cell in confusion_matrix[i]:
                msg += "\t" + str(cell)
            msg += "\n"
        for emotion in self.emotions:
            msg += "\t" + emotion
        msg += "\n\n"
        return msg


