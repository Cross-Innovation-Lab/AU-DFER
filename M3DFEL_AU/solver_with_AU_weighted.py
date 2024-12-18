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
            weight.append([x*5 for x in [0.2565426020973117, 0.22590006587965014, 0.7487900008174104, 0.24260938453624706, 0.3297348731445176, 0.6253054547733351, 0.2331741603449138, 0.6002659975024218, 0.25865288585343615, 0.4042681727124247, 0.25714836786909934, 0.4073109215604269, 0.22487756342746285, 0.1940789790636582, 0.7184070119769908, 0.6300831522957303, 0.4089852072091214, 0.22674904330132986]])
        elif emo==5:#disgust
            weight.append([x*5 for x in [0.3669101379226728, 0.23849898584920362, 0.8556853679596857, 0.2093396686826401, 0.47998741884320045, 0.9116481111605065, 0.47437419451562296, 0.8176370658093514, 0.3690013619929339, 0.6122010138157353, 0.37313028939087656, 0.43845703949339593, 0.2712229960450541, 0.22790281312128532, 0.7303843653361992, 0.5736046807175323, 0.4089852072091214, 0.3157383637072392]])
        elif emo==6:#fear
            weight.append([x*5 for x in [0.4668184913800416, 0.3236830048705496, 0.7092147921039609, 0.3237692334141919, 0.284442322105034, 0.4833791428919572, 0.19268179747966238, 0.49211626418097015, 0.29857493916455136, 0.40402139627383793, 0.3114334189399665, 0.42236998344528387, 0.2762397282204001, 0.20086024967771943, 0.6377967064381651, 0.6643128457350147, 0.4089852072091214, 0.22004117429592432]])
        elif emo==0:#happy
            weight.append([x*5 for x in [0.37458275378581574, 0.37673577978763106, 0.41551698634677164, 0.20406588498862172, 0.7927884197377446, 0.8938573401735845, 0.26152800062149906, 0.8999413926399066, 0.9306407335956202, 0.8244182955003968, 0.2344366715203141, 0.35584905526535116, 0.2637127706449251, 0.1980659294341865, 0.8749731256118369, 0.5415473658691405, 0.4089852072091214, 0.29454183039387216]])
        elif emo==2:#neutral
            weight.append([x*5 for x in [0.29634994040489937, 0.26249346997682327, 0.5349197805247116, 0.211357886947774, 0.24956316484226185, 0.48303878792569305, 0.1750459267403643, 0.4222786033590969, 0.2595297603319664, 0.4038707629862569, 0.2255822802606922, 0.3300116586908106, 0.21700897117154114, 0.18584396273490125, 0.4406741647399928, 0.4892561535112399, 0.4089852072091214, 0.24616267045793816]])
        elif emo==1:#sad
            weight.append([x*5 for x in [0.3563166809248465, 0.22025123080620432, 0.7883122756737404, 0.18552393229679323, 0.4330479071201498, 0.6512463687571445, 0.19562608246203084, 0.57228053429373, 0.3919835645137292, 0.5539327475091499, 0.26436050797140115, 0.3859719011102806, 0.2624203556065227, 0.19099832378533962, 0.5784889747902296, 0.5465085543388307, 0.4089852072091214, 0.31238729511297314]])
        elif emo==4:#surprise
            weight.append([x*5 for x in [0.45840639969725916, 0.4348135864186588, 0.5807746998255159, 0.32444326825812997, 0.27370636575780566, 0.49552874169247974, 0.22029419679136286, 0.45675173141757053, 0.30048086399990936, 0.35356016873960344, 0.2340917014067623, 0.35356016873960344, 0.22418778976044848, 0.18774040869030503, 0.6204770513319532, 0.6347324688342945, 0.4089852072091214, 0.2326942881893769]])
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


