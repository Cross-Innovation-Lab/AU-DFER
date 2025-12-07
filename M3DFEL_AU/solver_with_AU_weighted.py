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
    for line in target:
        emo = line.argmax()
        if emo==3:#angry
            weight.append([x*5 for x in [0.16978330014266482, 0.1474484099458089, 0.6385375278419455, 0.15955087911488458, 0.2257386547042889, 0.4972455283651611, 0.15269462267081732, 0.470889570126452, 0.1713443932394683, 0.28682403662377504, 0.17023111167232707, 0.28941226670462944, 0.14671370568908396, 0.12489530852760639, 0.6019091910383833, 0.5023565940498623, 0.2908397350173472, 0.1480589426871144]])
        elif emo==5:#disgust
            weight.append([x*5 for x in [0.2556613250379876, 0.15655681634966592, 0.7784678175136164, 0.13563151267233145, 0.353602732625092, 0.8594560438530944, 0.3484770561236273, 0.7265667054811852, 0.2573762546624107, 0.4833627557878272, 0.26077234063862126, 0.3163550480772833, 0.1807057096038953, 0.1488894096862477, 0.6161944289089377, 0.44359769238463886, 0.2908397350173472, 0.21474224698158992]])
        elif emo==6:#fear
            weight.append([x*5 for x in [0.3416233199942143, 0.22096625856984287, 0.591078697660434, 0.2210340665261555, 0.19066749128518232, 0.35671400783212986, 0.1239196009045801, 0.3647782154323708, 0.2014525981038694, 0.28661445976904537, 0.21138892966648018, 0.30233592318202696, 0.18447197078258057, 0.12964812274717674, 0.5106650931015151, 0.5397724065827655, 0.2908397350173472, 0.1432476989807878]])
        elif emo==0:#happy
            weight.append([x*5 for x in [0.26197021302251783, 0.2637489312360733, 0.2964310301204501, 0.13190483696240127, 0.6939537051210115, 0.8330800031639156, 0.17347610820526463, 0.8420321208563623, 0.8882934142237333, 0.7356394920285989, 0.15360866990050534, 0.24664740708336055, 0.17509971698083543, 0.1276861973423668, 0.8057327549388998, 0.41178926421868306, 0.2908397350173472, 0.19836039970638378]])
        elif emo==2:#neutral
            weight.append([x*5 for x in [0.19974525492944142, 0.17419319637385966, 0.4053456334801515, 0.13706230525786608, 0.16464150662864904, 0.3564013123982043, 0.1117065857762667, 0.30225692349830036, 0.17199394819786643, 0.28648655803768563, 0.14721999753281687, 0.22595757296022984, 0.14108230584951664, 0.11916179838996314, 0.31830472814177574, 0.36213049142283893, 0.2908397350173472, 0.16214811486455724]])
        elif emo==1:#sad
            weight.append([x*5 for x in [0.2470265618083506, 0.1433979246927842, 0.6881822893577378, 0.11893982182697348, 0.31161634476423544, 0.5253221085983302, 0.12597712051107948, 0.4422623779963743, 0.27645211415344356, 0.42395131504088984, 0.17558170449960314, 0.27142133890425624, 0.1741388696479932, 0.1227455638528141, 0.4485394778155102, 0.41664204112281555, 0.2908397350173472, 0.2121307850723057]])
        elif emo==4:#surprise
            weight.append([x*5 for x in [0.3340537870187493, 0.31316038897162674, 0.4508609549567622, 0.2215642997098194, 0.18256790904316805, 0.36794740171201545, 0.14342865607335295, 0.3325723523253354, 0.20291790550542987, 0.24479399356303455, 0.15335881131831128, 0.24479399356303455, 0.14621845929676547, 0.12047847974103684, 0.4921072671708563, 0.5073560806954985, 0.2908397350173472, 0.15234747176484023]])
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



