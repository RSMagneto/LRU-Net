import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import CBAM
import scipy
from torch.nn import init
# from einops import repeat
import scipy.io as sio
from torch.autograd import Variable

class CNNLayer(nn.Module):
    def __init__(self):
        super(CNNLayer, self).__init__()
        self.CBAM = CBAM(gate_channels=64)
        device = torch.device('cuda:0')
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=31, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        ).to(device)
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        ).to(device)
    def forward(self, LRHS, HRMS):
        LRHS_att = self.conv1(LRHS)
        HRMS_att = self.conv2(HRMS)
        attention_spe, attention_spa = self.CBAM(LRHS_att, HRMS_att)
        return attention_spe, attention_spa

class sparse(nn.Module):
    def __init__(self):
        super(sparse, self).__init__()
        device = torch.device('cuda:0')
        self.CNNLayer = CNNLayer()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=31, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        ).to(device)
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=31, kernel_size=3, stride=1, padding=0)
        ).to(device)
    def forward(self, input_shape, weight, Upre, Z, LRHS, HRMS):
        self.nb, self.nt, self.nx, self.ny = input_shape
        LRHS = LRHS.reshape(self.nb, self.nt, self.nx, self.ny)
        Upre = Upre.reshape(self.nb, self.nt, self.nx, self.ny)
        Z = Z.reshape(self.nb, self.nt, self.nx, self.ny)
        attention_spe, attention_spa = self.CNNLayer(LRHS, HRMS)
        Z = self.conv1(Z)
        mid = Z * attention_spe * attention_spa + Z
        mid = self.conv2(mid)
        out = LRHS - mid + Upre / weight
        out = torch.reshape(out, [self.nb, self.nt, self.nx * self.ny])
        return out

class lowrank(nn.Module):
    def __init__(self):
        super(lowrank, self).__init__()
        self.thres_coef = Variable(torch.tensor(-2, dtype=torch.float32), requires_grad=True)
    def forward(self, weight, Upre, Vpre, Dpre, Tpre, LRHS):
        Z = (1 / (weight + weight)) * (Upre + Vpre + weight * (LRHS - Dpre) + weight * Tpre)
        Ut, St, Vt = torch.linalg.svd(Z, full_matrices=False)
        thres = torch.sigmoid(self.thres_coef) * St[:, 0]
        thres = torch.unsqueeze(thres, -1)
        St = nn.functional.relu(St - thres)
        St = torch.diag_embed(St)
        US = torch.matmul(Ut, St)
        ZA = torch.matmul(US, Vt)
        return ZA
class dataconsis(nn.Module):
    def __init__(self):
        super(dataconsis, self).__init__()
    def forward(self, alpha, weight, HRMS, SR, Vpre, Z):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        raw_int_bs = list(HRMS.size())
        b, c, w, h = raw_int_bs[0], raw_int_bs[1], raw_int_bs[2], raw_int_bs[3]
        SST = alpha * torch.mm(SR, SR.permute(1, 0)) + weight * torch.eye(SR.shape[0]).to(device)
        SST_inv = torch.inverse(SST)
        out = []
        for i in range(HRMS.shape[0]):
            x = HRMS[i].squeeze(0)
            x = x.reshape(c, w * h).permute(1, 0)
            x1 = torch.mm(x, SR.permute(1, 0))
            mid = alpha * x1 - Vpre[i].permute(1, 0) + weight * Z[i].permute(1, 0)
            out1 = torch.mm(mid, SST_inv)
            out1 = out1.view(1, w * h, -1).permute(0, 2, 1)
            out.append(out1)
        out = torch.cat(out, 0)
        return out


class LplusS_Net(nn.Module):
    def __init__(self, niter=8, alpha=256, beta=2):
        super(LplusS_Net, self).__init__()
        self.niter = niter
        self.alpha = alpha
        self.b = beta
        self.sparse = sparse()
        self.lowrank = lowrank()
        self.dataconsis = dataconsis()
        self.conv_cat = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=248, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_lrhs = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=31, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_hrms = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=31, kernel_size=3, stride=1, padding=0),
        )

    def forward(self, LRHS, HRMS, SR):
        LRHS = F.interpolate(LRHS, scale_factor=4, mode='bicubic', align_corners=True)
        lrhs = LRHS
        nb, nt, nx, ny = lrhs.shape
        LRHS = torch.reshape(lrhs, [nb, nt, nx * ny])
        Tpre = torch.reshape(lrhs, [nb, nt, nx * ny])
        Upre = torch.zeros_like(Tpre)
        Vpre = torch.zeros_like(Tpre)
        Dpre = torch.zeros_like(Tpre)
        weight = 0.1
        fea_cat = []
        for i in range(self.niter):
            Z = self.lowrank(weight, Upre, Vpre, Dpre, Tpre, LRHS)
            D = self.sparse(lrhs.shape, weight, Upre, Z, LRHS, HRMS)
            T = self.dataconsis(self.alpha, weight, HRMS, SR, Vpre, Z)
            U1 = Upre + weight * (LRHS - Z - D)
            V1 = Vpre + weight * (T - Z)
            Upre = U1
            Vpre = V1
            Dpre = D
            Tpre = T
            weight = self.b * weight
            fea_cat.append(Z)
        fea_cat = torch.cat(fea_cat, dim=1)
        fea_cat = torch.reshape(fea_cat, [nb, fea_cat.shape[1], nx, ny])
        fea = self.conv_cat(fea_cat)
        lrhs = self.conv_lrhs(lrhs)
        hrms = self.conv_hrms(HRMS)
        out = fea + lrhs + hrms
        out = self.conv(out)
        return out
