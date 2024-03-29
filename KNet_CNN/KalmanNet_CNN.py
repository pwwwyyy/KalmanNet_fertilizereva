import torch
import torch.nn as nn
import torch.nn.functional as func


class KalmanCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def NNBuild(self, h, m, n, t_fa, v_fa, sw, args):

        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.m = m
        self.h = h
        self.n = n
        self.t_fa = t_fa
        self.v_fa = v_fa
        self.sw = sw
        self.batch_size = args.n_batch  # Batch size
        self.InitCNN(args)

    def InitCNN(self, args):
        self.conv_x_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.m * 2, out_channels=16, kernel_size=2, stride=1),  # sw - kernel_size +1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)  # -kernel_size+1
        ).to(self.device)

        self.conv_x_2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1),  # sw - kernel_size +1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)  # -kernel_size+1
        ).to(self.device)

        self.conv_x_3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1),  # sw - kernel_size +1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)  # -kernel_size+1
        ).to(self.device)

        self.conv_x_4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=1),  # sw - kernel_size +1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)  # -kernel_size+1
        ).to(self.device)

        self.conv_x_5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=1),  # sw - kernel_size +1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)  # -kernel_size+1
        ).to(self.device)

        self.conv_y_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.n * 2, out_channels=4, kernel_size=2, stride=1),  # sw - kernel_size +1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)  # -kernel_size+1
        ).to(self.device)

        self.conv_y_2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=2, stride=1),  # sw - kernel_size +1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)  # -kernel_size+1
        ).to(self.device)

        self.conv_y_3 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2, stride=1),  # sw - kernel_size +1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)  # -kernel_size+1
        ).to(self.device)

        self.conv_y_4 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1),  # sw - kernel_size +1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)  # -kernel_size+1
        ).to(self.device)

        self.conv_y_5 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1),  # sw - kernel_size +1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)  # -kernel_size+1
        ).to(self.device)

        self.d_input_FC0 = 256
        self.d_output_FC0 = self.m ** 2
        self.FC0 = nn.Sequential(
            nn.Linear(self.d_input_FC0, self.d_output_FC0),
            nn.ReLU()).to(self.device)

        self.d_input_FC1 = 64
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.ReLU()).to(self.device)

        self.d_input_FC2 = self.n ** 2 + self.m ** 2
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2)).to(self.device)


    # def InitSequence(self, inputy, target, T):
    #     """
    #        (torch.tensor): 1st moment of x at time sw [batch_size, m/n]
    #     """
    #     self.T = T
    #     self.m1x_posterior = target[:, :, self.sw-1].to(self.device)
    #     self.m1x_posterior_previous = target[:, :, self.sw-2].to(self.device)
    #     i = self.sw-2
    #     F = torch.tensor([[1, -(self.t_fa[i + 1001] - self.t_fa[i + 1000]),
    #                        -self.v_fa[i + 1000] * (self.t_fa[i + 1001] - self.t_fa[i + 1000]),
    #                        -self.v_fa[i + 1000] ** 2 * (self.t_fa[i + 1001] - self.t_fa[i + 1000])],
    #                       [0, 1, 0, 0],
    #                       [0, 0, 1, 0],
    #                       [0, 0, 0, 1]]).float()
    #     batched_F = F.view(1, F.shape[0], F.shape[1]).expand(target.shape[0], -1, -1).to(self.device)
    #     self.m1x_prior_previous = torch.squeeze(torch.bmm(batched_F, torch.unsqueeze(target[:, :, i - 2], dim=2)), dim=2).to(self.device)
    #     self.y_previous = inputy[:, :, self.sw-1].to(self.device)

    def InitSequence(self, inputy, T):
        self.T = T
        self.m1x_posterior = self.x_posterior_sw0[:, :, self.sw - 1].to(self.device)
        self.m1x_posterior_previous = self.x_posterior_sw0[:, :, self.sw - 2].to(self.device)
        self.m1x_prior_previous = self.x_prior_sw0[:, :, self.sw - 1].to(self.device)
        self.y_previous = inputy[:, :, self.sw - 1].to(self.device)


    def InitSW(self, inputy, target, m2x_1000, H_onlyPos, Q_gen, R_onlyPos):  # batchsize,m/n，T  , m2x_1000, H_onlyPos, Q_gen, R_onlyPos
        # # both in size [batch_size, n/m]
        # self.y_sw_obs_diff = torch.zeros(inputy.shape[0], inputy.shape[1], self.sw).to(self.device)
        # self.y_sw_obs_diff[:, :, 0] = func.normalize((inputy[:, :, 0]-inputy[:, :, 0]), p=2, dim=0, eps=1e-12, out=None).to(self.device)  # dim = ? #############################################
        #
        # self.y_sw_obs_innov_diff = torch.zeros(inputy.shape[0], inputy.shape[1], self.sw).to(self.device)
        # self.y_sw_obs_innov_diff[:, :, 0] = func.normalize((inputy[:, :, 0] - inputy[:, :, 0]), p=2, dim=0, eps=1e-12, out=None).to(self.device)
        #
        # self.x_sw_fw_evol_diff = torch.zeros(target.shape[0], target.shape[1], self.sw).to(self.device)
        # self.x_sw_fw_evol_diff[:, :, 0] = func.normalize((target[:, :, 0] - target[:, :, 0]), p=2, dim=0, eps=1e-12, out=None).to(self.device)
        #
        # self.x_sw_fw_update_diff = torch.zeros(target.shape[0], target.shape[1], self.sw).to(self.device)
        # self.x_sw_fw_update_diff[:, :, 0] = func.normalize((target[:, :, 0] - target[:, :, 0]), p=2, dim=0, eps=1e-12, out=None).to(self.device)
        #
        # for i in range(1, self.sw):
        #     F = torch.tensor([[1, -(self.t_fa[i + 1001] - self.t_fa[i + 1000]),
        #                        -self.v_fa[i + 1000] * (self.t_fa[i + 1001] - self.t_fa[i + 1000]),
        #                        -self.v_fa[i + 1000] ** 2 * (self.t_fa[i + 1001] - self.t_fa[i + 1000])],
        #                       [0, 1, 0, 0],
        #                       [0, 0, 1, 0],
        #                       [0, 0, 0, 1]]).float()
        #     batched_F = F.view(1, F.shape[0], F.shape[1]).expand(target.shape[0], -1, -1).to(self.device)  # expand -1 不改变当前维度
        #     m1x_prior = torch.bmm(batched_F, torch.unsqueeze(target[:, :, i-1], dim=2)).to(self.device)  # batchsize, m ,1
        #     m1y = self.h(m1x_prior).to(self.device)
        #     m1y = torch.squeeze(m1y, dim=2)  # batchsize,n
        #     self.y_sw_obs_diff[:, :, i] = func.normalize((inputy[:, :, i] - inputy[:, :, i - 1]), p=2, dim=0, eps=1e-12, out=None).to(self.device)
        #     self.y_sw_obs_innov_diff[:, :, i] = func.normalize((inputy[:, :, i] - m1y), p=2, dim=0, eps=1e-12, out=None).to(self.device)
        #     if i == 1:
        #         self.x_sw_fw_evol_diff[:, :, i] = self.x_sw_fw_evol_diff[:, :, 0].to(self.device)
        #         self.x_sw_fw_update_diff[:, :, i] = self.x_sw_fw_update_diff[:, :, 0].to(self.device)
        #     else:
        #         m1x_prior_previous = torch.bmm(batched_F, torch.unsqueeze(target[:, :, i-2], dim=2)).to(self.device)
        #         m1x_prior_previous = torch.squeeze(m1x_prior_previous, dim=2)
        #         self.x_sw_fw_evol_diff[:, :, i] = func.normalize((target[:, :, i-1] - target[:, :, i - 2]), p=2, dim=0, eps=1e-12, out=None).to(self.device)
        #         self.x_sw_fw_update_diff[:, :, i] = func.normalize((target[:, :, i-1] - m1x_prior_previous), p=2, dim=0, eps=1e-12, out=None).to(self.device)


        # both in size [batch_size, n/m]
        self.y_sw_obs_diff = torch.zeros(inputy.shape[0], inputy.shape[1], self.sw).to(self.device)
        self.y_sw_obs_diff[:, :, 0] = func.normalize((inputy[:, :, 0]-inputy[:, :, 0]), p=2, dim=0, eps=1e-12, out=None).to(self.device)  # dim = ? #############################################

        self.y_sw_obs_innov_diff = torch.zeros(inputy.shape[0], inputy.shape[1], self.sw).to(self.device)
        self.y_sw_obs_innov_diff[:, :, 0] = func.normalize((inputy[:, :, 0] - inputy[:, :, 0]), p=2, dim=0, eps=1e-12, out=None).to(self.device)

        self.x_sw_fw_evol_diff = torch.zeros(target.shape[0], target.shape[1], self.sw).to(self.device)
        self.x_sw_fw_evol_diff[:, :, 0] = func.normalize((target[:, :, 0] - target[:, :, 0]), p=2, dim=0, eps=1e-12, out=None).to(self.device)

        self.x_sw_fw_update_diff = torch.zeros(target.shape[0], target.shape[1], self.sw).to(self.device)
        self.x_sw_fw_update_diff[:, :, 0] = func.normalize((target[:, :, 0] - target[:, :, 0]), p=2, dim=0, eps=1e-12, out=None).to(self.device)

        self.x_prior_sw0 = torch.zeros(target.shape[0], target.shape[1], self.sw).to(self.device)
        self.x_posterior_sw0 = torch.zeros(target.shape[0], target.shape[1], self.sw).to(self.device)
        self.x_prior_sw0[:, :, 0] = target[:, :, 0].to(self.device)
        self.x_posterior_sw0[:, :, 0] = target[:, :, 0].to(self.device)
        m2x_0_batch = m2x_1000.view(1, self.m, self.m).expand(target.shape[0], -1, -1)
        m2x_posterior = m2x_0_batch.to(self.device)
        batched_H = H_onlyPos.view(1, H_onlyPos.shape[0], H_onlyPos.shape[1]).expand(target.shape[0], -1, -1).to(self.device)
        batched_H_T = torch.transpose(batched_H, 1, 2).to(self.device)
        for i in range(1, self.sw):
            F = torch.tensor([[1, -(self.t_fa[i + 1001] - self.t_fa[i + 1000]),
                               -self.v_fa[i + 1000] * (self.t_fa[i + 1001] - self.t_fa[i + 1000]),
                               -self.v_fa[i + 1000] ** 2 * (self.t_fa[i + 1001] - self.t_fa[i + 1000])],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]]).float()
            batched_F = F.view(1, F.shape[0], F.shape[1]).expand(target.shape[0], -1, -1).to(self.device)  # expand -1 不改变当前维度
            batched_F_T = torch.transpose(batched_F, 1, 2).to(self.device)
            m1x_prior = torch.bmm(batched_F, torch.unsqueeze(self.x_posterior_sw0[:, :, i-1], dim=2)).to(self.device)  # batchsize, m ,1
            yt = torch.unsqueeze(inputy[:, :, i], 2)
            m1y = self.h(m1x_prior).to(self.device)
            m2x_prior = torch.bmm(batched_F, m2x_posterior)
            m2x_prior = torch.bmm(m2x_prior, batched_F_T) + Q_gen.to(self.device)
            m2y = torch.bmm(batched_H, m2x_prior)
            m2y = torch.bmm(m2y, batched_H_T) + R_onlyPos.to(self.device)
            KG = torch.bmm(m2x_prior, batched_H_T)
            KG = torch.bmm(KG, torch.inverse(m2y))
            dy = yt - m1y
            # Compute the 1-st posterior moment
            m1x_posterior = m1x_prior + torch.bmm(KG, dy)
            # Compute the 2-nd posterior moment
            m2x_posterior = torch.bmm(m2y, torch.transpose(KG, 1, 2))
            m2x_posterior = m2x_prior - torch.bmm(KG, m2x_posterior)
            self.x_posterior_sw0[:, :, i] = torch.squeeze(m1x_posterior, dim=2)
            self.x_prior_sw0[:, :, i] = torch.squeeze(m1x_prior, dim=2)

            self.y_sw_obs_diff[:, :, i] = func.normalize((inputy[:, :, i] - inputy[:, :, i - 1]), p=2, dim=0, eps=1e-12, out=None).to(self.device)
            self.y_sw_obs_innov_diff[:, :, i] = func.normalize((inputy[:, :, i] - torch.squeeze(m1y, dim=2)), p=2, dim=0, eps=1e-12, out=None).to(self.device)
            if i == 1:
                self.x_sw_fw_evol_diff[:, :, i] = self.x_sw_fw_evol_diff[:, :, 0].to(self.device)
                self.x_sw_fw_update_diff[:, :, i] = self.x_sw_fw_update_diff[:, :, 0].to(self.device)
            else:
                self.x_sw_fw_evol_diff[:, :, i] = func.normalize((self.x_posterior_sw0[:, :, i-1] - self.x_posterior_sw0[:, :, i - 2]), p=2, dim=0, eps=1e-12, out=None).to(self.device)
                self.x_sw_fw_update_diff[:, :, i] = func.normalize((self.x_posterior_sw0[:, :, i-1] - self.x_prior_sw0[:, :, i-1]), p=2, dim=0, eps=1e-12, out=None).to(self.device)



    def KGain_step(self, y_sw, x_sw):
        x_out_conv1 = self.conv_x_1(x_sw)
        x_out_conv2 = self.conv_x_2(x_out_conv1)
        x_out_conv3 = self.conv_x_3(x_out_conv2)
        x_out_conv4 = self.conv_x_4(x_out_conv3)
        x_out_conv5 = self.conv_x_5(x_out_conv4)
        y_out_conv1 = self.conv_y_1(y_sw)
        y_out_conv2 = self.conv_y_2(y_out_conv1)
        y_out_conv3 = self.conv_y_3(y_out_conv2)
        y_out_conv4 = self.conv_y_4(y_out_conv3)
        y_out_conv5 = self.conv_y_5(y_out_conv4)
        x_out_conv = torch.reshape(x_out_conv5, (self.batch_size, 256))  # sw = 11
        y_out_conv = torch.reshape(y_out_conv5, (self.batch_size, 64))
        x_out_FC0 = self.FC0(x_out_conv)
        y_out_FC1 = self.FC1(y_out_conv)
        out_FC2 = self.FC2(torch.cat((x_out_FC0, y_out_FC1), 1))
        return out_FC2

    def forward(self, y, t):  # y=test_input[:,:, t] (batchsize,n)
        y = y.to(self.device)

        F = torch.tensor([[1, -(self.t_fa[t + 1001] - self.t_fa[t + 1000]),
                           -self.v_fa[t + 1000] * (self.t_fa[t + 1001] - self.t_fa[t + 1000]),
                           -self.v_fa[t + 1000] ** 2 * (self.t_fa[t + 1001] - self.t_fa[t + 1000])],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]).float()
        batched_F = F.to(self.m1x_posterior.device).view(1, F.shape[0], F.shape[1]).expand(self.m1x_posterior.shape[0], -1, -1)
        self.m1x_prior = torch.bmm(batched_F, torch.unsqueeze(self.m1x_posterior, 2))
        self.m1y = self.h(self.m1x_prior)
        self.m1x_prior = torch.squeeze(self.m1x_prior, 2)
        self.m1y = torch.squeeze(self.m1y, 2)
        # both in size [batch_size, n]
        obs_diff = y - self.y_previous
        obs_innov_diff = y - self.m1y
        # both in size [batch_size, m]
        fw_evol_diff = self.m1x_posterior - self.m1x_posterior_previous
        fw_update_diff = self.m1x_posterior - self.m1x_prior_previous
        obs_diff = func.normalize(obs_diff, p=2, dim=0, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=0, eps=1e-12, out=None)
        # updating sliding window  ###
        for i in range(0, self.sw-1):
            self.y_sw_obs_diff[:, :, i] = self.y_sw_obs_diff[:, :, i+1]
            self.y_sw_obs_innov_diff[:, :, i] = self.y_sw_obs_innov_diff[:, :, i+1]
            self.x_sw_fw_evol_diff[:, :, i] = self.x_sw_fw_evol_diff[:, :, i+1]
            self.x_sw_fw_update_diff[:, :, i] = self.x_sw_fw_update_diff[:, :, i+1]

        self.y_sw_obs_diff[:, :, self.sw-1] = obs_diff
        self.y_sw_obs_innov_diff[:, :, self.sw-1] = obs_innov_diff
        self.x_sw_fw_evol_diff[:, :, self.sw-1] = fw_evol_diff
        self.x_sw_fw_update_diff[:, :, self.sw-1] = fw_update_diff

        y_sw = torch.cat((self.y_sw_obs_diff, self.y_sw_obs_innov_diff), dim=1)
        x_sw = torch.cat((self.x_sw_fw_evol_diff, self.x_sw_fw_update_diff), dim=1)

        # Kalman Gain Network Step
        KG = self.KGain_step(y_sw, x_sw)
        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))
        # Innovation
        dy = torch.unsqueeze(y, dim=2) - torch.unsqueeze(self.m1y, dim=2)   # [batch_size, n, 1]
        INOV = torch.bmm(self.KGain, dy)
        INOV = torch.squeeze(INOV,dim=2)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV  # [batch_size, m]
        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y

        return self.m1x_posterior
