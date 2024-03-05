import torch
import torch.nn as nn
import torch.nn.functional as func


class DoubleConv(torch.nn.Module):
    def __init__(self, channel_in, channel_out, device):
        super(DoubleConv, self).__init__()
        self.device = device
        self.dcl = nn.Sequential(
            nn.Conv1d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(channel_out),
            nn.ReLU(),

            nn.Conv1d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(channel_out),
            nn.ReLU(),
        ).to(self.device)

    def forward(self, x):
        x.to(self.device)
        return self.dcl(x)


class DownSampling(torch.nn.Module):
    def __init__(self, device):
        super(DownSampling, self).__init__()
        self.device = device
        self.Down = nn.MaxPool1d(kernel_size=2, stride=2, padding=0).to(self.device)

    def forward(self, x):
        x.to(self.device)
        return self.Down(x)


class UpSampling(torch.nn.Module):
    def __init__(self, channel_in, channel_out, device):
        super(UpSampling, self).__init__()
        self.device = device
        self.Up = nn.ConvTranspose1d(in_channels=channel_in, out_channels=channel_out, kernel_size=2, stride=2).to(self.device)

    def forward(self, x):
        x.to(self.device)
        return self.Up(x)


class OutConv(torch.nn.Module):
    def __init__(self, channel_in, channel_out, device):
        super(OutConv, self).__init__()
        self.device = device
        self.out = nn.Conv1d(in_channels=channel_in, out_channels=channel_out, kernel_size=1).to(self.device)

    def forward(self, x):
        x.to(self.device)
        return self.out(x)


class UNet(torch.nn.Module):
    def __init__(self, channel, device):
        super(UNet, self).__init__()
        self.device = device

        self.convleft1 = DoubleConv(channel_in=channel, channel_out=channel*8, device=self.device)
        self.convleft2 = DoubleConv(channel_in=channel*8, channel_out=channel*16, device=self.device)
        self.convleft3 = DoubleConv(channel_in=channel*16, channel_out=channel*32, device=self.device)
        self.convleft4 = DoubleConv(channel_in=channel*32, channel_out=channel*64, device=self.device)
        self.convleft5 = DoubleConv(channel_in=channel*64, channel_out=channel*128, device=self.device)
        self.ds = DownSampling(self.device)
        self.up1 = UpSampling(channel_in=channel*128, channel_out=channel*64, device=self.device)
        self.up2 = UpSampling(channel_in=channel*64, channel_out=channel*32, device=self.device)
        self.up3 = UpSampling(channel_in=channel*32, channel_out=channel*16, device=self.device)
        self.up4 = UpSampling(channel_in=channel*16, channel_out=channel*8, device=self.device)
        self.convright1 = DoubleConv(channel_in=channel*128, channel_out=channel*64, device=self.device)
        self.convright2 = DoubleConv(channel_in=channel*64, channel_out=channel*32, device=self.device)
        self.convright3 = DoubleConv(channel_in=channel*32, channel_out=channel*16, device=self.device)
        self.convright4 = DoubleConv(channel_in=channel*16, channel_out=channel*8, device=self.device)
        self.out = OutConv(channel_in=channel*8, channel_out=channel, device=self.device)

    def forward(self, x):

        x.to(self.device)
        out_cl1 = self.convleft1(x).to(self.device)
        ds_cl1 = self.ds(out_cl1).to(self.device)
        out_cl2 = self.convleft2(ds_cl1).to(self.device)
        ds_cl2 = self.ds(out_cl2).to(self.device)
        out_cl3 = self.convleft3(ds_cl2).to(self.device)
        ds_cl3 = self.ds(out_cl3).to(self.device)
        out_cl4 = self.convleft4(ds_cl3).to(self.device)
        # out_up1 = self.up1(out_cl4).to(self.device)
        # crop_cl3 = out_cl3
        # out_cr1 = self.convright1(torch.cat((crop_cl3, out_up1), dim=1)).to(self.device)
        # out_up2 = self.up2(out_cr1).to(self.device)
        # crop_cl2 = out_cl2
        # out_cr2 = self.convright2(torch.cat((crop_cl2, out_up2), dim=1)).to(self.device)
        # out_up3 = self.up3(out_cr2).to(self.device)
        # crop_cl1 = out_cl1
        # out_cr3 = self.convright3(torch.cat((crop_cl1, out_up3), dim=1)).to(self.device)
        ds_cl4 = self.ds(out_cl4).to(self.device)
        out_cl5 = self.convleft5(ds_cl4).to(self.device)
        out_up1 = self.up1(out_cl5).to(self.device)
        crop_cl4 = out_cl4
        out_cr1 = self.convright1(torch.cat((crop_cl4, out_up1), dim=1)).to(self.device)
        out_up2 = self.up2(out_cr1).to(self.device)
        crop_cl3 = out_cl3
        out_cr2 = self.convright2(torch.cat((crop_cl3, out_up2), dim=1)).to(self.device)
        out_up3 = self.up3(out_cr2).to(self.device)
        crop_cl2 = out_cl2
        out_cr3 = self.convright3(torch.cat((crop_cl2, out_up3), dim=1)).to(self.device)
        out_up4 = self.up4(out_cr3).to(self.device)
        crop_cl1 = out_cl1
        out_cr4 = self.convright4(torch.cat((crop_cl1, out_up4), dim=1)).to(self.device)

        # batchsize m sw
        out = self.out(out_cr4).to(self.device)
        return out


class KF_UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def NNBuild(self, m, n, args):

        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.m = m
        self.n = n
        self.batch_size = args.n_batch  # Batch size
        self.unet = UNet(channel=1, device=self.device)

    def InitSequence(self,  T):
        self.T = T

    # def KFTest(self, test_input):
    #     # # allocate memory for KF output
    #     # KF_out = torch.zeros(self.batch_size, self.m, self.T).to(self.device)
    #     #
    #     # m1x_0_batch = self.m1x_1000.view(1, self.m, 1).expand(self.batch_size, -1, -1).to(self.device)
    #     # m2x_1000 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()
    #     # m2x_0_batch = m2x_1000.view(1, self.m, self.m).expand(self.batch_size, -1, -1).to(self.device)
    #     #
    #     # test_input = test_input.to(self.device)
    #     #
    #     # def batched_F(f):
    #     #     return f.view(1, self.m, self.m).expand(self.batch_size, -1, -1).to(self.device)
    #     #
    #     # def batched_F_T(f):
    #     #     return torch.transpose(batched_F(f), 1, 2).to(self.device)
    #     #
    #     # H_onlyPos = torch.tensor([[1, 0, 0, 0]]).float()
    #     # Q_gen = torch.tensor([[1, 0, 0, 0], [0, 0.001, 0, 0], [0, 0, 0.000001, 0], [0, 0, 0, 0.000001]]).float()
    #     # R_gen = torch.tensor([500]).float()
    #     # R_onlyPos = R_gen
    #     # batched_H = H_onlyPos.view(1, self.n, self.m).expand(self.batch_size, -1, -1).to(self.device)
    #     # batched_H_T = torch.transpose(batched_H, 1, 2).to(self.device)
    #     # # Allocate Array for 1st and 2nd order moments (use zero padding)
    #     # x = torch.zeros(self.batch_size, self.m, self.T).to(self.device)
    #     # sigma = torch.zeros(self.batch_size, self.m, self.m, self.T).to(self.device)
    #     # # Set 1st and 2nd order moments for t=0
    #     # m1x_posterior = m1x_0_batch.to(self.device)
    #     # m2x_posterior = m2x_0_batch.to(self.device)
    #     # # Generate in a batched manner
    #     # for t in range(0, self.T):
    #     #     F_gen = torch.tensor([[1, -(self.t_fa[t + 1001] - self.t_fa[t + 1000]),
    #     #                            -self.v_fa[t + 1000] * (self.t_fa[t + 1001] - self.t_fa[t + 1000]),
    #     #                            -self.v_fa[t + 1000] ** 2 * (self.t_fa[t + 1001] - self.t_fa[t + 1000])],
    #     #                           [0, 1, 0, 0],
    #     #                           [0, 0, 1, 0],
    #     #                           [0, 0, 0, 1]]).float()
    #     #     yt = torch.unsqueeze(test_input[:, :, t], 2)
    #     #     m1x_prior = torch.bmm(batched_F(F_gen), m1x_posterior).to(self.device)
    #     #     # Predict the 2-nd moment of x
    #     #     m2x_prior = torch.bmm(batched_F(F_gen), m2x_posterior)
    #     #     m2x_prior = torch.bmm(m2x_prior, batched_F_T(F_gen)) + Q_gen.to(self.device)
    #     #     # Predict the 1-st moment of y
    #     #     m1y = torch.bmm(batched_H, m1x_prior)
    #     #     # Predict the 2-nd moment of y
    #     #     m2y = torch.bmm(batched_H, m2x_prior)
    #     #     m2y = torch.bmm(m2y, batched_H_T) + R_onlyPos.to(self.device)
    #     #     KG = torch.bmm(m2x_prior, batched_H_T)
    #     #     KG = torch.bmm(KG, torch.inverse(m2y))
    #     #     dy = yt - m1y
    #     #     # Compute the 1-st posterior moment
    #     #     m1x_posterior = m1x_prior + torch.bmm(KG, dy)
    #     #     # Compute the 2-nd posterior moment
    #     #     m2x_posterior = torch.bmm(m2y, torch.transpose(KG, 1, 2))
    #     #     m2x_posterior = m2x_prior - torch.bmm(KG, m2x_posterior)
    #     #     xt = m1x_posterior
    #     #     sigmat = m2x_posterior
    #     #     x[:, :, t] = torch.squeeze(xt, 2)
    #     #     sigma[:, :, :, t] = sigmat
    #     # KF_out = x
    #     return KF_out

    def forward(self, y):
        y = y.to(self.device)
        CNN_out = torch.zeros(y.shape[0], 1, y.shape[2]).to(self.device)  # batchsize m T
        CNN_out[:, 0, :] = y[:, 0, :]
        norm = torch.norm(CNN_out, p=2, dim=0)
        CNN_out = func.normalize(CNN_out, p=2, dim=0, eps=1e-12, out=None)    ##################      dim=2?
        unet_out = self.unet(CNN_out)
        for i in range(0, unet_out.shape[1]):
            for j in range(0, unet_out.shape[2]):
                unet_out[:, i, j] = unet_out[:, i, j] * norm[i, j]
        return unet_out


