
from torchattacks import *
import torchattacks
import foolbox as fb
import argparse
import torch
import copy
import dataset
import numpy as np
import adv_attacks
import os
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import torchvision
import auc_utils
from auc_utils import *
# import det_network
# import detector_net
import detector_net_inference

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='cifar10|cifar100|TinyImagenet')
parser.add_argument('--a_type', default='fgsm')
parser.add_argument('--batch_size', type=int, default='200')
parser.add_argument('--energy_path', default='.')
parser.add_argument('--class_model_pth', default='.')
parser.add_argument('--model_adv_gen', default='.')
parser.add_argument('--model_inference', default='.')
parser.add_argument('--lut_path', default='.')
parser.add_argument('--clean', type=int, default=1)
parser.add_argument('--baseline', type=int, default=0)
parser.add_argument('--calc_err', type=int, default=0)
parser.add_argument('--save_energy_dist', type=int, default=0)
parser.add_argument('--pgd_param', default='0.125,0.007,7')
parser.add_argument('--cw_param', default='0,1e-4,100')
parser.add_argument('--aname', default='fgsm')
parser.add_argument('--percentile', type=int, default=0)
parser.add_argument('--n_batches', type=int, default=0)
parser.add_argument('--stage', type=int, default=2)
parser.add_argument('--sles', default='vgg16', help='the SLes model')
parser.add_argument('--model', default='vgg16', help='the SV model')
parser.add_argument('--file_task', default='vgg16', help='the SV model')

args = parser.parse_args()

if args.type == 'cifar10':
    print('loading cifar10 dataset')
    train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1, train= True)
    # import cifar10_net
    # import cifar10_net_infer

    # model = cifar10_net.Net().cuda()
    if args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    if args.model == 'vgg16':
        model = torchvision.models.vgg16(pretrained=False, num_classes=10)
    if args.model == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=False, num_classes=10)
    if args.model == 'vgg19':
        model = torchvision.models.vgg19(pretrained=False, num_classes=10)

    model = torch.nn.DataParallel(model)

    try:
        model.load_state_dict(torch.load(args.model_adv_gen).state_dict())
    except:
        model.load_state_dict(torch.load(args.model_adv_gen)['state_dict'])

    model_infer = detector_net_inference.Net()

    model_infer = torch.nn.DataParallel(model_infer)

    try:
        model_infer.load_state_dict(torch.load(args.model_inference).state_dict())
    except:
        model_infer.load_state_dict(torch.load(args.model_inference))
    model = model.cuda()
    model_infer = model_infer.cuda()

    if args.calc_err == 1:
        model_mc = torchvision.models.vgg16(pretrained=False, num_classes=10)
        model_mc = torch.nn.DataParallel(model_mc)

        try:
            model_mc.load_state_dict(torch.load(args.class_model_pth).state_dict())
        except:
            model_mc.load_state_dict(torch.load(args.class_model_pth)['state_dict'])
        model_mc.eval()
        model_mc = model_mc.cuda()


elif args.type == 'cifar100':  #### change dataset to CIFAR100 if b present
    print('loading cifar100 dataset')
    # train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1, train=True)
    train_loader, test_loader = dataset.get100(batch_size=args.batch_size)

    # print('using the VGG19 network')
    if args.model=='resnet18':
        model = torchvision.models.resnet18(pretrained=False, num_classes=100)
    if args.model == 'vgg16':
        model = torchvision.models.vgg16(pretrained=False, num_classes=100)
    if args.model == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=False, num_classes=100)
    if args.model == 'vgg19':
        model = torchvision.models.vgg19(pretrained=False, num_classes=100)
    # model = models.resnet18(pretrained=False, num_classes=100)
    # model = models.vgg16_bn() #models.resnet18(pretrained=False, num_classes=100) #vgg19_bn() #models.vgg19_bn() #tinyimagenet_net.ResNet().cuda()
    # model.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
    #                                nn.ReLU(inplace=True), nn.Dropout(p=0.5),
    #                                nn.Linear(in_features=512, out_features=256, bias=True),
    #                                nn.ReLU(inplace=True), nn.Dropout(p=0.5),
    #                                nn.Linear(in_features=256, out_features=100, bias=True))
    # model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    model = torch.nn.DataParallel(model)
    print(model)
    try:
        model.load_state_dict(torch.load(args.model_adv_gen).state_dict())
    except:
        model.load_state_dict(torch.load(args.model_adv_gen)['state_dict'])

    model_infer = detector_net_inference.Net()
    # model_infer.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
    #                                nn.ReLU(inplace=True), nn.Dropout(p=0.5),
    #                                nn.Linear(in_features=512, out_features=256, bias=True),
    #                                nn.ReLU(inplace=True), nn.Dropout(p=0.5),
    #                                nn.Linear(in_features=256, out_features=100, bias=True)) #cifar100_net_infer.Net().cuda()
    model_infer = torch.nn.DataParallel(model_infer)
    try:
        model_infer.load_state_dict(torch.load(args.model_inference).state_dict())
    except:
        model_infer.load_state_dict(torch.load(args.model_inference))
    model = model.cuda()
    model_infer = model_infer.cuda()

    if args.calc_err == 1:
        model_mc = torchvision.models.mobilenet_v2(pretrained=False, num_classes=100)
        model_mc = torch.nn.DataParallel(model_mc)

        try:
            model_mc.load_state_dict(torch.load(args.class_model_pth).state_dict())
        except:
            model_mc.load_state_dict(torch.load(args.class_model_pth)['state_dict'])
        model_mc.eval()
        model_mc = model_mc.cuda()

elif args.type == 'tinyimagenet':
    print('loading tinyimagenet dataset')
    train_loader, test_loader = dataset.tinyimagenet(batch_size=args.batch_size)
    if args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False, num_classes=200)
    if args.model == 'vgg16':
        model = torchvision.models.vgg16(pretrained=False, num_classes=200)
    if args.model == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=False, num_classes=200)
    if args.model == 'vgg19':
        model = torchvision.models.vgg19(pretrained=False, num_classes=200)

    model = torch.nn.DataParallel(model)

    try:
        model.load_state_dict(torch.load(args.model_adv_gen).state_dict())
    except:
        model.load_state_dict(torch.load(args.model_adv_gen)['state_dict'])

    model_infer = detector_net_inference.Net()

    model_infer = torch.nn.DataParallel(model_infer)

    try:
        model_infer.load_state_dict(torch.load(args.model_inference).state_dict())
    except:
        model_infer.load_state_dict(torch.load(args.model_inference))
    model = model.cuda()
    model_infer = model_infer.cuda()

    if args.calc_err == 1:
        model_mc = torchvision.models.vgg16(pretrained=False, num_classes=200)
        model_mc = torch.nn.DataParallel(model_mc)

        try:
            model_mc.load_state_dict(torch.load(args.class_model_pth).state_dict())
        except:
            model_mc.load_state_dict(torch.load(args.class_model_pth)['state_dict'])
        model_mc.eval()
        model_mc = model_mc.cuda()

if args.a_type == 'fgsm' and args.clean == 0:
    eps,alpha,steps = map(float, args.pgd_param.split(','))
    print(f'FGSM with eps = {eps}')
    energy_path = args.energy_path + '/energy_' + args.type + '_' + args.a_type + '_e=' + str(eps)
if args.a_type == 'cw' and args.clean == 0:
    kappa,c,steps = map(float, args.cw_param.split(','))
    steps = int(steps)
    attack = torchattacks.CW(model, c= c, kappa=kappa, steps=steps)
    # print(f'FGSM with eps = {eps}')
    inp_energy_path = args.energy_path + '/energy_mod_' + args.type + '_' + args.a_type
if args.a_type == 'pgd' and args.clean == 0:
    eps,alpha,steps = map(float, args.pgd_param.split(','))
    steps = int(steps)
    # model.eval()
    if args.aname == 'gn':
        attack = GN(model, sigma=0.1)
    elif args.aname == 'autoattack':
        attack = AutoAttack(model, eps=16/255, n_classes=100, version='standard')
    elif args.aname == 'apgd':
        attack = APGD(model, eps=8/255, steps=2, eot_iter=100, n_restarts=1, loss='ce')
    elif args.aname == 'pgd4':
        attack = attack = torchattacks.PGD(model, eps=0.02, alpha=0.008, steps=10, random_start=True)
    elif args.aname == 'pgd2':
        attack = attack = torchattacks.PGD(model, eps=0.01, alpha=0.004, steps=10, random_start=True)
    elif args.aname == 'pgd1':
        attack = attack = torchattacks.PGD(model, eps=0.005, alpha=0.002, steps=10, random_start=True)
    elif args.aname == 'pgd8':
        attack = attack = torchattacks.PGD(model, eps=0.04, alpha=0.016, steps=10, random_start=True)
    elif args.aname == 'pgd16':
        attack = attack = torchattacks.PGD(model, eps=0.08, alpha=0.04, steps=10, random_start=True)
    elif args.aname == 'tpgd':
        attack = TPGD(model, eps=8 / 255., alpha=2 / 255., steps=10)
    elif args.aname == 'square':
        attack = Square(model, eps=0.3, n_queries=10, n_restarts=1, loss='ce', p_init=0.8)
    elif args.aname == 'cw':
        attack = CW(model, c=100, lr=0.1, steps=100, kappa=0)
    elif args.aname == 'pgdl2':
        attack = torchattacks.PGDL2(model, eps=5.0, alpha=0.5, steps=50, random_start=True)
    elif args.aname == 'difgsm':
        attack = DIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=10, diversity_prob=0.5, resize_rate=0.9)
    elif args.aname == 'mifgsm':
        attack = MIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=10, decay=0.1)
    elif args.aname == 'fab':
        attack = torchattacks.FAB(model, norm='Linf', steps=100, eps=0.25, n_restarts=1, alpha_max=0.1, eta=1.05,
                                  beta=0.9, verbose=False, seed=0, targeted=False, n_classes=100)
    elif args.aname == 'bim':
        attack = torchattacks.BIM(model, eps=8 / 255., alpha=2 / 255., steps=10)  # , random_start=True)
    elif args.aname == 'ffgsm':
        attack = FFGSM(model, eps=8 / 255., alpha=2 / 255.)
    elif args.aname == 'fgsm':
        attack = FGSM(model, eps=8 / 255.)
    # attack = torchattacks.PGD(model, eps=4 / 255, alpha=1 / 255, steps=10, random_start=True)
    #
    # print("Torchattacks", torchattacks.__version__)
    #
    # attack = DeepFool(model, steps=1, overshoot=0.08)
    #
    #
    # attack = OnePixel(model, pixels=5, inf_batch=500)
    #
    # attack = APGDT(model, eps=4/255, steps=10, eot_iter=1, n_restarts=1)
    #
    #

    #

    #
    # attack = torchattacks.FGSM(model, eps=8/255.)
    # attack = fa.LinfPGD()
    print(f'PGD with eps = {eps}, alpha = {alpha}, steps = {steps}')
    energy_path = args.energy_path + '/energy_' + args.type + '_' + args.a_type+'_e=' + str(eps)+'_a=' + str(alpha)+'_n=' + str(steps)
    # inp_soi_path = args.soi_path + '/inp_soi_sles_'+args.sles+'_sv_'+args.model+'_' + args.type + '_' + args.aname+'_e=' + str(eps)+'_a=' + str(alpha)+'_n=' + str(steps)

if args.clean == 1:
    print(f'Using clean inputs')
    energy_path = args.energy_path + '/energy_' + args.type + '_clean'
    inp_energy_path = args.energy_path + '/inp_energy_sles_'+args.sles+'_'+args.type + '_clean'
if args.clean == 2:
    print(f'Using clean sample inputs')
    test_loader = train_loader
    # soi_path = args.soi_path + '/soi_' + args.type + '_clean'
    inp_energy_path = args.energy_path + '/inp_energy_' + args.type + '_snat'

n_batches = int(args.n_batches)
print(n_batches)
esnat = []
for i, (data, labels_tru) in enumerate(train_loader):

    if i < n_batches:
        print(data.size())
        inp_adv = data.cuda()
        with torch.no_grad():
            energy_snat = model_infer(inp_adv, args.stage)
        tuple_da_inp = (energy_snat.cpu(), labels_tru.cpu())
        esnat.append(tuple_da_inp)
    else:
        break
print(len(esnat))
arr_snat = list_to_np_arr(esnat)
per = np.percentile(arr_snat, args.percentile)
print(per)

e_nat = []
for i, (data, labels_tru) in enumerate(test_loader):
    inp_clean = data.cuda()
    with torch.no_grad():
        energy_nat = model_infer(inp_clean, args.stage)
    tuple_da_inp = (energy_nat.cpu(), labels_tru.cpu())
    e_nat.append(tuple_da_inp)

if args.save_energy_dist == 1:
    nat_energy_path = args.energy_path+'/natural_stage_'+str(args.stage)
    print(f'saving energy at : {nat_energy_path}')
    # torch.save(soi_list, soi_path)
    torch.save(e_nat, nat_energy_path)
arr_nat = list_to_np_arr(e_nat)

count = 0
adv_data_batch = []
adv_label_batch = []
adv_test_set = []
energy_list = []
correct = 0
error = 0
accuracy = 0
access_list = []
inp_energy_list = []
model_infer.eval()
model.train()
for i, (data, labels_tru) in enumerate(test_loader):
    print(i)
    indx_target = labels_tru.cpu().clone()
    if args.clean == 0:
        if args.a_type == 'pgd':
            # fmodel = fb.PyTorchModel(model, bounds=(0, 1))
            # inp_adv = attack(fmodel, data.cuda(), labels_tru.cuda(), epsilons=[0.1])
            # inp_adv = inp_adv[0]
            # inp_adv = data.cuda() + torch.normal(mean=1, std=0.1, size=data.size()).cuda()
            inp_adv = attack(data.cuda(), labels_tru.cuda()) #adv_attacks.pgd_attack(model, data.cuda(), labels_tru.cuda(), eps, alpha, steps )
        elif args.a_type == 'fgsm':
            inp_adv = adv_attacks.fgsm_attack(data, eps, data_grad)
        elif args.a_type == 'cw':
            inp_adv = attack(data.cuda(), labels_tru.cuda())


    elif args.clean == 1:
        inp_adv = data.cuda()
    elif args.clean == 2:
        inp_adv = data.cuda()
    with torch.no_grad():
        energy_2 = model_infer(inp_adv, args.stage)
    # inp_soi = inp_adv.abs().mean(dim=1).mean(dim=1).mean(dim=1)
    # x = inp_adv
    # with torch.no_grad():
    #     for i in range(8):
    #         # print('hi')
    #         x = model_infer.module.features[i](x)
    #
    #     soi_2 = x.abs().mean(dim=1).mean(dim=1).mean(dim=1)
    # soi = model_infer.module.features[0](inp_adv).abs().mean(dim=1).mean(dim=1).mean(dim=1)

    if args.calc_err == 1:
        if args.baseline == 0:
            for idx, si in enumerate(energy_2):
                if si <= per:
                    adv_sample = 0

                else:
                    adv_sample = 1

                # rand_no = np.random.random_sample()
                if args.clean == 1 and adv_sample == 0:
                    output = model_mc(inp_adv)
                    pred = torch.argmax(output[idx,:].data)
                    accuracy += pred.cpu().eq(indx_target[idx]).sum()
                if args.clean == 0 and adv_sample == 0:
                    output = model_mc(inp_adv)
                    pred = torch.argmax(output[idx,:].data)
                    error += 1-pred.cpu().eq(indx_target[idx]).sum()

        else:
            if args.clean == 1:
                output = model_mc(inp_adv)
                pred = output.data.max(1)[1]
                accuracy += pred.cpu().eq(indx_target).sum()
            elif args.clean == 0:
                output = model_mc(inp_adv)
                pred = output.data.max(1)[1]
                error += (1-(pred.cpu().eq(indx_target)).int()).sum()

    # tuple_da = (soi, labels_tru)
    tuple_da_inp = (energy_2.cpu(), labels_tru.cpu())

    # soi_list.append(tuple_da)
    # print(i)
    if args.clean == 2 and i < 50:
        inp_energy_list.append(tuple_da_inp)
    elif args.clean == 2 and i >= 50:
        break
    else:
        inp_energy_list.append(tuple_da_inp)
arr_adv = list_to_np_arr(inp_energy_list)
auc = calc_auc(arr_nat, arr_adv, per)
print(f'baseline: {args.baseline}, attack: {args.aname}, accuracy: {accuracy}, error: {error}, auc: {auc}')
print(f' log file path {args.file_task}')
if os.path.exists(args.file_task) == 0:
    f = open(args.file_task, 'w')
else:
    f = open(args.file_task, 'a')
f.write(f'{auc}, ')
# f.write(f'baseline: {args.baseline}, attack: {args.aname}, accuracy: {accuracy}, error: {error}, auc: {auc} \n')
f.close()
if args.save_energy_dist == 1:
    inp_energy_path = args.energy_path+'/adv_stage_'+str(args.stage)
    print(f'saving energy_dist at : {inp_energy_path}')
    # torch.save(soi_list, soi_path)
    torch.save(inp_energy_list, inp_energy_path)
else:
    print('energy_dist not saved')


