import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.models.vgg import vgg16


def l2_loss(output, target):
    return nn.MSELoss()(output, target)

def nll_loss(output, target):
    return F.nll_loss(output, target)

#light smooth loss


def gradient(input_tensor, direction):
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
    smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)

    if direction == 'x':
        kernel = smooth_kernel_x
    else:
        kernel = smooth_kernel_y

    grad = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
    grad_min = torch.min(grad)
    grad_max = torch.max(grad)
    grad_norm = torch.div((grad - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


def smooth_light_loss(low_light, high_light):
    low_grad_x = gradient(low_light, 'x')
    high_grad_x = gradient(high_light, 'x')
    x_loss = (low_grad_x + high_grad_x) * torch.exp(-10 * (low_grad_x + high_grad_x))
    low_grad_y = gradient(low_light, 'y')
    high_grad_y = gradient(high_light, 'y')
    y_loss = (low_grad_y + high_grad_y) * torch.exp(-10 * (low_grad_y + high_grad_y))
    xy_loss = torch.mean(x_loss + y_loss)
    return xy_loss


def smooth_light_input_loss(light, input):
    gray_input = rgb_to_grayscale(input)
    low_grad_x = gradient(light, 'x')
    high_grad_x = gradient(gray_input, 'x')
    x_loss = torch.abs(torch.div(low_grad_x, torch.clamp(high_grad_x, min=0.01)))
    low_grad_y = gradient(light, 'y')
    high_grad_y = gradient(gray_input, 'y')
    y_loss = torch.abs(torch.div(low_grad_y, torch.clamp(high_grad_y, min=0.01)))
    xy_loss = torch.mean(x_loss + y_loss)
    return xy_loss


def ave_gradient(input_tensor, direction):
    return F.avg_pool2d(gradient(input_tensor, direction),
                        kernel_size=3, stride=1, padding=1)


def smooth_I_R(I, R):
    R = rgb_to_grayscale(R)
    I = rgb_to_grayscale(I)
    return torch.mean(gradient(R, "x") * torch.exp(-10 * ave_gradient(I, "x")) +
                      gradient(R, "y") * torch.exp(-10 * ave_gradient(I, "y")))


def l1_loss(input, target):
    return nn.L1Loss()(input, target)


def L_color(x, y):
    mean_rgb_x = torch.mean(x, [2, 3], keepdim=True)
    mean_rgb_y = torch.mean(y, [2, 3], keepdim=True)
    mr_x, mg_x, mb_x = torch.split(mean_rgb_x, 1, dim=1)
    mr_y, mg_y, mb_y = torch.split(mean_rgb_y, 1, dim=1)
    Dr = torch.pow(mr_x - mr_y, 2)
    Db = torch.pow(mb_x - mb_y, 2)
    Dg = torch.pow(mg_x - mg_y, 2)
    output = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)

    return torch.mean(output)


class gray_color(nn.Module):

    def __init__(self):
        super(gray_color, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return torch.mean(k)


class R_I_color(nn.Module):

    def __init__(self):
        super(R_I_color, self).__init__()

    def forward(self, x, y):
        mean_rgbx = torch.mean(x, [2, 3], keepdim=True)
        mean_rgby = torch.mean(y, [2, 3], keepdim=True)
        mrx, mgx, mbx = torch.split(mean_rgbx, 1, dim=1)
        mry, mgy, mby = torch.split(mean_rgby, 1, dim=1)
        Drgx = l1_loss(mrx, mgx)
        Drbx = l1_loss(mrx, mbx)
        Dgbx = l1_loss(mbx, mgx)
        Drgy = l1_loss(mry, mgy)
        Drby = l1_loss(mry, mby)
        Dgby = l1_loss(mby, mgy)

        k = torch.abs(Drgx - Drgy) + torch.abs(Drbx - Drby) + torch.abs(Dgbx - Dgby)

        return k


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3
