import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
import pickle
import torch.backends.cudnn as cudnn


def train_class_batch(model, samples, target, criterion, AU_labels,dataset,posw_option, AU_ratio):
    outputs,AU_outputs = model(samples)
    loss = criterion(outputs, target)
    
    AU_cri = AU_cri_set(target,dataset,posw_option)
    AU_loss = AU_cri(AU_outputs,AU_labels)
    loss = loss*(1 - AU_ratio) + AU_loss * AU_ratio
    return loss, outputs

def vec_mul(a,b):
    res = []
    for i in range(0,len(a)):
        res.append(a[i]*b[i])
    return res

def AU_cri_set(target,dataset,posw_option):
    print(dataset)
    weight = []
    pos_weight = []
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
    #print('weight size',weight.size())
    if posw_option == 'global' and dataset == 'DFEW':
        pos_weight = np.array([8.123028391167193, 10.40521645603657, 1.0299431287937402, 1.904973346878674, 4.991242525542634, 2.9253478113335594, 31.506833567505428, 2.493520755545794, 7.428694442604491, 2.5988969808385773, 4.979137299126022, 2.8861470803811384, 12.160409556313994, 6.5054854311666865, 5.102436217149434, 9.670691823899372, 101.28938906752411, 8.483380533611566])
    if posw_option == 'global' and dataset == 'FERV39k':
        pos_weight = np.array([4.789480561907873, 4.554315802670344, 1.4431454725929194, 1.2930490140261892, 10.91287980639957, 2.960842161727236, 12.93426639408712, 4.830240821160679, 9.943312337902928, 3.558962749536942, 5.529216712106698, 3.1841620626151017, 10.882526485181709, 8.180273518441775, 3.816699282452707, 5.920877919237679, 128.54385964912282, 5.021610601427115])
    if posw_option == 'distinct' and dataset == 'DFEW':
        for line in target:
            emo = line.argmax()
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
    if posw_option == 'distinct' and dataset == 'FERV39k':
        for line in target:
            emo = line.argmax()
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
    if posw_option == 'minor' and dataset == 'DFEW':
        for line in target:
            emo = line.argmax()
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
    if posw_option == 'minor' and dataset == 'FERV39k':
        for line in target:
            emo = line.argmax()
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

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, Dataset='DFER', posw_option='global',AU_ratio = 0.25):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _,AU_labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        AU_labels = AU_labels.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, AU_labels, Dataset, posw_option, AU_ratio)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion, AU_labels, Dataset, posw_option, AU_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        #uar = np.mean(class_acc)
        #war = conf_mat.trace() / conf_mat.sum()
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(loss=loss_value)
        #metric_logger.update(uar=uar)
        #metric_logger.update(war=war)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, Dataset='DFER', posw_option='global',AU_ratio = 0.25):
    criterion = torch.nn.CrossEntropyLoss()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    outputs, targets = [], []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        AU_labels = batch[-1]
        AU_labels = AU_labels.to(device, non_blocking=True)
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            #output = model(videos)
            output,AU_outputs = model(videos)
            loss = criterion(output, target)
            AU_cri = AU_cri_set(target,Dataset,posw_option)
            AU_loss = AU_cri(AU_outputs,AU_labels)
            loss = loss*(1-AU_ratio) + AU_loss * AU_ratio
            #loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        output, target = output.cpu().detach().numpy(), target.cpu().detach().numpy()
        outputs.append(output)
        targets.append(target)

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # me: cal total metrics across the val set
    preds, labels = np.concatenate(outputs), np.concatenate(targets)
    preds = np.argmax(preds, axis=1)
    from sklearn.metrics import confusion_matrix, f1_score
    conf_mat = confusion_matrix(y_pred=preds, y_true=labels)
    class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
    uar = np.mean(class_acc)
    war = conf_mat.trace() / conf_mat.sum()
    weighted_f1 = f1_score(y_pred=preds, y_true=labels, average='weighted')
    micro_f1 = f1_score(y_pred=preds, y_true=labels, average='micro')
    macro_f1 = f1_score(y_pred=preds, y_true=labels, average='macro')
    metric_logger.meters['uar'].update(uar, n=len(preds))
    metric_logger.meters['war'].update(war, n=len(preds))
    metric_logger.meters['weighted_f1'].update(weighted_f1, n=len(preds))
    metric_logger.meters['micro_f1'].update(micro_f1, n=len(preds))
    metric_logger.meters['macro_f1'].update(macro_f1, n=len(preds))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    print('* WAR {war.global_avg:.4f} UAR {uar.global_avg:.4f} weighted_f1 {weighted_f1.global_avg:.4f} micro_f1 {micro_f1.global_avg:.4f} macro_f1 {macro_f1.global_avg:.4f}'
          .format(war=metric_logger.war, uar=metric_logger.uar, weighted_f1=metric_logger.weighted_f1, micro_f1=metric_logger.micro_f1, macro_f1=metric_logger.macro_f1))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def final_test(data_loader, model, device, file, save_feature=False,Dataset='DFER', posw_option='global',AU_ratio = 0.25):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    # me: for saving feature in the last layer
    saved_features = {}

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        AU_labels = batch[5]
        AU_labels = AU_labels.to(device, non_blocking=True)
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            # me: for saving feature in the last layer
            output,AU_outputs = model(videos)
            #loss = criterion(output, target)
            AU_cri = AU_cri_set(target,Dataset,posw_option)
            if save_feature:
                output, saved_feature ,AU_outputs = model(videos, save_feature=save_feature)
            else:
                output,AU_outputs = model(videos)
                #loss = criterion(output, target)
                AU_loss = AU_cri(AU_outputs,AU_labels)
            loss = criterion(output, target)
            AU_loss = AU_cri(AU_outputs,AU_labels)
            loss = loss*(1-AU_ratio) + AU_loss * AU_ratio

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

            # me: for saving feature in the last layer
            if save_feature:
                if ids[i] not in saved_features:
                    saved_features[ids[i]] = {'chunk_id': [], 'split_id': [],
                                              'label': int(target[i].cpu().numpy()),
                                              'feature': [], 'logit': []}
                saved_features[ids[i]]['chunk_id'].append(int(chunk_nb[i].cpu().numpy()))
                saved_features[ids[i]]['split_id'].append(int(split_nb[i].cpu().numpy()))
                saved_features[ids[i]]['feature'].append(saved_feature.data[i].cpu().numpy().tolist())
                saved_features[ids[i]]['logit'].append(output.data[i].cpu().numpy().tolist())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)

    # me: for saving feature in the last layer
    if save_feature:
        feature_file = file.replace(file[-4:], '_feature.pkl')
        pickle.dump(saved_features, open(feature_file, 'wb'))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    #return {k: metric_logger.meters['uar']*0.5+metric_logger.meters['war']*0.5 for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks, args, best=False):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    # me: for saving feature in the last layer
    overall_saved_features = {}

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt') if not best else os.path.join(eval_path, str(x) + '_best.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float64, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label

        # me: for saving feature in the last layer
        if args.save_feature:
            feature_file = file.replace(file[-4:], '_feature.pkl')
            saved_features = pickle.load(open(feature_file, 'rb'))
            for sample_id in saved_features.keys():
                if sample_id not in overall_saved_features:
                    overall_saved_features[sample_id] = {
                        'chunk_split_id': [], # the only identifier for each view
                        'label': saved_features[sample_id]['label'],
                        'feature': [], 'prob': []}
                chunk_ids = saved_features[sample_id]['chunk_id']
                split_ids = saved_features[sample_id]['split_id']
                for idx, (chunk_id, split_id) in enumerate(zip(chunk_ids, split_ids)):
                    chunk_split_id = f"{chunk_id}_{split_id}"
                    # avoid repetition
                    if chunk_split_id not in overall_saved_features[sample_id]['chunk_split_id']:
                        overall_saved_features[sample_id]['chunk_split_id'].append(chunk_split_id)
                        overall_saved_features[sample_id]['feature'].append(saved_features[sample_id]['feature'][idx])
                        # NOTE: do softmax, logit -> prob
                        overall_saved_features[sample_id]['prob'].append(softmax(saved_features[sample_id]['logit'][idx]))


    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    # me: more metrics and save preds
    pred_dict = {'id': [], 'label': [], 'pred': []}
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
        pred = int(np.argmax(np.mean(dict_feats[item], axis=0)))
        label = int(dict_label[item])
        pred_dict['pred'].append(pred)
        pred_dict['label'].append(label)
        pred_dict['id'].append(item.strip())
    # from multiprocessing import Pool
    # p = Pool(4)
    # ans = p.map(compute_video, input_lst)
    # me: disable multi-process because it often gets stuck
    ans = [compute_video(lst) for lst in input_lst]
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)

    # me: for saving feature in the last layer
    if args.save_feature:
        # get avg feature and pred
        for sample_id in overall_saved_features.keys():
            overall_saved_features[sample_id]['feature'] = np.mean(overall_saved_features[sample_id]['feature'], axis=0)
            overall_saved_features[sample_id]['pred'] = int(np.argmax(np.mean(overall_saved_features[sample_id]['prob'], axis=0)))
        feature_file = os.path.join(eval_path, 'overall_feature.pkl') if not best else os.path.join(eval_path, 'overall_feature_best.pkl')
        pickle.dump(overall_saved_features, open(feature_file, 'wb'))

    return final_top1*100 ,final_top5*100, pred_dict

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
