import torch
import torch.nn.functional as F
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import copy
from torch.autograd import Variable


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    if epsilon!=0:
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def pgd_attack(net, inp_var, true_label, epsilon, alpha, num_steps):
    step = 0
    inp_var = Variable(inp_var, requires_grad=True)
    inp_adv = inp_var
    inp_adv = Variable(inp_adv, requires_grad=True)

    while step < num_steps:
        output = net(inp_var)
        loss_ad = F.cross_entropy(output, true_label)
        loss_ad.backward()

        inp_adv = inp_adv + alpha * torch.sign(inp_var.grad.data)
        eta = inp_adv - inp_var
        eta = eta.clamp(-epsilon, epsilon)
        inp_adv.data = inp_var + eta
        step += 1

    return inp_adv.detach()