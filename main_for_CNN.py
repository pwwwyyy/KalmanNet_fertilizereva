import torch
from datetime import datetime
from torch.distributions.multivariate_normal import MultivariateNormal
import config as config
import time
import random
import torch.nn as nn
import numpy as np
from KNet_UNet.KalmanNet_UNet import KalmanUNet
from Pipeline.Pipeline_CNN import Pipeline_CNN as Pipeline_CNN
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1E4
import matplotlib.pyplot as plt

N_fa = 4000
t_fa = [0]
for i in range(1, N_fa+1):
    # t_fa.append(t_fa[-1] + random.uniform(0.9, 1.1))
    t_fa.append(t_fa[-1] + 1)


def v_est(t):
    if 1000 < t < 3000:
        return 10 * abs(np.sin(t / (50 * np.pi)))
    else:
        return 0


v = np.array([v_est(ti) for ti in t_fa], dtype=np.float64)
v_fa = torch.from_numpy(v)

m = 4
n = 1

#########################################################
### state evolution matrix F and observation matrix H ###
#########################################################

H_onlyPos = torch.tensor([[1, 0, 0, 0]]).float()
###############################################
### process noise Q and observation noise R ###
###############################################
# Noise Parameters
# Q_gen = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()
Q_gen = torch.tensor([[1, 0, 0, 0], [0, 0.001, 0, 0], [0, 0, 0.000001, 0], [0, 0, 0, 0.000001]]).float()
# R_onlyPos = torch.tensor([10000]).float()
R_gen = torch.tensor([500]).float()
R_onlyPos = R_gen
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
print("Pipeline Start")

####################################
### Generative Parameters For CA ###
####################################
args = config.general_settings()
### Dataset parameters
args.N_E = 1000  #'input training dataset size (# of sequences)' 100, m ,4000
args.N_CV = 100  #input cross validation dataset size (# of sequences)
args.N_T = 100   #input test dataset size (# of sequences)
args.randomInit_train = False
args.randomInit_cv = False
args.randomInit_test = False

args.T = 100  # input sequence length
args.T_test = 100  # input test sequence length
### training parameters
args.use_cuda = True # use GPU or not
args.n_steps = 4000  # number of training steps
args.n_batch = 16  #input batch size for training
args.lr = 1e-4
args.wd = 1e-4
args.out_mult_KNet = 32

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

m1x_1000 = torch.tensor([[19000], [1], [0.5], [0.1]]).float()
m2x_1000 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()
sw = 11

#############################
###  Dataset Generation   ###
#############################

Loss_On_AllState = False  # if false: only calculate loss on position
Train_Loss_On_AllState = True  # if false: only calculate training loss on position

######################
### Generate Batch ###
######################
# Allocate Empty Array for Input

def GenerateBatch( args, size, T, randomInit=False):  # size=N_E and so on tranning size
    # fixed init
    initConditions = m1x_1000.view(1, m, 1).expand(size, -1,
                                                    -1)  # view重新定义矩阵形状为1 m 1 （改变前后元素个数相同）expand扩展为更大维度
    ### for sequence generation
    m1x_0_batch = initConditions
    x_prev = m1x_0_batch
    m2x_0_batch = m2x_1000

    # Allocate Empty Array for Input
    Input = torch.empty(size, n, T)
    # Allocate Empty Array for Target
    Target = torch.empty(size, m, T)

    # Set x0 to be x previous
    x_prev = m1x_0_batch
    xt = x_prev
    # Generate in a batched manner
    for t in range(0, T):
            ########################
            #### State Evolution ###
            ########################
            # F_gen = torch.tensor([[1, -1, 0, 0],
            #                       [0, 1, 0, 0],
            #                       [0, 0, 1, 0],
            #                       [0, 0, 0, 1]]).float()
            F_gen = torch.tensor([[1, -(t_fa[t + 1001] - t_fa[t+1000]),
                                   -v_fa[t+1000] * (t_fa[t + 1001] - t_fa[t+1000]),
                                   -v_fa[t+1000] ** 2 * (t_fa[t + 1001] - t_fa[t+1000])],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]).float()
            # No noise 比较两个张量是否相同，此处如果Q=0
            batched_F = F_gen.to(x_prev.device).view(1, F_gen.shape[0], F_gen.shape[1]).expand(x_prev.shape[0], -1, -1)
            xt = torch.bmm(batched_F, x_prev)
            mean = torch.zeros([size, m])
            distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
            eq = distrib.rsample().view(size, m, 1)
            # Additive Process Noise
            xt = torch.add(xt, eq)
            # 相当于生成真实轨迹##################这里参数变化时需要调整########################

            ################
            ### Emission ###
            ################
            # Observation Noise
            def h_b(x):
                batched_H = H_onlyPos.to(x.device).view(1, H_onlyPos.shape[0], H_onlyPos.shape[1]).expand(x.shape[0], -1, -1)
                return torch.bmm(batched_H, x)
            if torch.equal(R_gen, torch.zeros(n, n)):  # No noise
                yt = h_b(xt)
            elif n == 1:  # 1 dim noise
                yt = h_b(xt)
                # er = torch.normal(mean=torch.zeros(size), std=R_gen).view(size, 1, 1)  # ##是否可以先用加性噪声训练###########################################################
                er = torch.normal(mean=torch.zeros(size), std=0.03).view(size, 1, 1)
                er = torch.mul(yt, er)
                # Additive Observation Noise
                yt = torch.add(yt, er)
            else:
                yt = H_onlyPos.matmul(xt)  ##类似矩阵相乘
                mean = torch.zeros([size, n])
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                er = distrib.rsample().view(size, n, 1)
                # Additive Observation Noise
                yt = torch.add(yt, er)  # 此处生成测量值

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            Target[:, :, t] = torch.squeeze(xt, 2)

            # Save Current Observation to Trajectory Array
            Input[:, :, t] = torch.squeeze(yt, 2)

            ################################
            ### Save Current to Previous ###
            ################################
            x_prev = xt
    return Input, Target, m1x_0_batch


DatafolderName = 'data_UNet/'
DatafileName = 'fertilizer_1000.pt'
print("Start Data Gen")
train_input, train_target, train_init = GenerateBatch(args, args.N_E, args.T, randomInit=args.randomInit_train)
cv_input, cv_target, cv_init = GenerateBatch(args, args.N_CV, args.T, randomInit=args.randomInit_cv)
test_input, test_target, test_init = GenerateBatch(args, args.N_T, args.T_test, randomInit=args.randomInit_test)
#save data
torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init], DatafolderName+DatafileName)
print("Load Original Data")
[train_input, train_target, cv_input, cv_target, test_input, test_target,train_init,cv_init,test_init] = torch.load(DatafolderName+DatafileName, map_location=device)
print("Data Shape")
print("testset state x size:", test_target.size())
print("testset observation y size:", test_input.size())
print("self.N_E in pipeline", len(train_input))
print("trainset state x size:", train_target.size())
print("trainset observation y size:", train_input.size())
print("cvset state x size:", cv_target.size())
print("cvset observation y size:", cv_input.size())


def h(x):
    batched_H = H_onlyPos.to(x.device).view(1, H_onlyPos.shape[0], H_onlyPos.shape[1]).expand(x.shape[0], -1, -1)
    return torch.bmm(batched_H, x)


# ##########################
# ### Evaluate KalmanNet ###
# ##########################
# Build Neural Network
KNet_model = KalmanUNet()
KNet_model.NNBuild(h, m, n, t_fa, v_fa, sw, args)
print("Number of trainable parameters for KNet pass 1:", sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
# Train Neural Network
KNet_Pipeline = Pipeline_CNN(strTime, "KNet")
KNet_Pipeline.setssModel(m, n, args.T, h, args.T_test, m2x_1000, H_onlyPos, Q_gen, R_onlyPos)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(args)
path_results = 'KNet_UNet/'
# [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState)
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_out, RunTime] = KNet_Pipeline.NNTest(test_input, test_target, path_results, MaskOnState=not Loss_On_AllState)  # , load_model=True, load_model_path=path_results + 'best-model-200c4f3,45db.pt'


def KFTest(args, test_input, test_target, allStates=True,\
     randomInit = False, test_init=None, test_lengthMask=None):
    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_KF_linear_arr = torch.zeros(args.N_T)
    # allocate memory for KF output
    KF_out = torch.zeros(args.N_T, m, args.T_test)
    if not allStates:
        loc = torch.tensor([True,False,False,False]) # for position only
        if m == 3:
            loc = torch.tensor([True,False,False]) # for position only

    start = time.time()

    m1x_0_batch = m1x_1000.view(1, m, 1).expand(args.N_T, -1, -1)
    m2x_0_batch = m2x_1000.view(1, m, m).expand(args.N_T, -1, -1)

    test_input = test_input.to(device)
    batch_size = test_input.shape[0]  # batch size
    T = test_input.shape[2]  # sequence length (maximum length if randomLength=True)
    def batched_F(f):
        return f.view(1, m, m).expand(batch_size, -1, -1).to(device)
    def batched_F_T(f):
        return torch.transpose(batched_F(f), 1, 2).to(device)

    batched_H = H_onlyPos.view(1, n, m).expand(batch_size, -1, -1).to(device)
    batched_H_T = torch.transpose(batched_H, 1, 2).to(device)
    # Allocate Array for 1st and 2nd order moments (use zero padding)
    x = torch.zeros(batch_size, m, T).to(device)
    KGout = torch.zeros(m, n, T).to(device)
    sigma = torch.zeros(batch_size, m, m, T).to(device)
    # Set 1st and 2nd order moments for t=0
    m1x_posterior = m1x_0_batch.to(device)
    m2x_posterior = m2x_0_batch.to(device)
    # Generate in a batched manner
    for t in range(0, T):
        # F_gen = torch.tensor([[1, -1, 0, 0],
        #                       [0, 1, 0, 0],
        #                       [0, 0, 1, 0],
        #                       [0, 0, 0, 1]]).float()
        F_gen = torch.tensor([[1, -(t_fa[t + 1001] - t_fa[t+1000]), -v_fa[t+1000] * (t_fa[t + 1001] - t_fa[t+1000]), -v_fa[t+1000] ** 2 * (t_fa[t + 1001] - t_fa[t+1000])],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]]).float()
        yt = torch.unsqueeze(test_input[:, :, t], 2)
        m1x_prior = torch.bmm(batched_F(F_gen), m1x_posterior).to(device)
        # Predict the 2-nd moment of x
        m2x_prior = torch.bmm(batched_F(F_gen), m2x_posterior)
        m2x_prior = torch.bmm(m2x_prior, batched_F_T(F_gen)) + Q_gen.to(device)
        # Predict the 1-st moment of y
        m1y = torch.bmm(batched_H, m1x_prior)
        # Predict the 2-nd moment of y
        m2y = torch.bmm(batched_H, m2x_prior)
        m2y = torch.bmm(m2y, batched_H_T) + R_onlyPos.to(device)
        KG = torch.bmm(m2x_prior, batched_H_T)
        KG = torch.bmm(KG, torch.inverse(m2y))
        dy = yt - m1y
        # Compute the 1-st posterior moment
        m1x_posterior = m1x_prior + torch.bmm(KG, dy)
        # Compute the 2-nd posterior moment
        m2x_posterior = torch.bmm(m2y, torch.transpose(KG, 1, 2))
        m2x_posterior = m2x_prior - torch.bmm(KG, m2x_posterior)
        xt = m1x_posterior
        sigmat = m2x_posterior
        x[:, :, t] = torch.squeeze(xt, 2)
        KGout[:, :, t] = KG[0]
        sigma[:, :, :, t] = sigmat
    end = time.time()
    t = end - start
    KF_out = x
    # MSE loss
    for j in range(args.N_T):  # cannot use batch due to different length and std computation
        if (allStates):
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(x[j, :, test_lengthMask[j]],
                                               test_target[j, :, test_lengthMask[j]]).item()
            else:
                MSE_KF_linear_arr[j] = loss_fn(x[j, :, :], test_target[j, :, :]).item()
        else:  # mask on state
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(x[j, loc, test_lengthMask[j]],
                                               test_target[j, loc, test_lengthMask[j]]).item()
            else:
                MSE_KF_linear_arr[j] = loss_fn(x[j, loc, :], test_target[j, loc, :]).item()

    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)
    print(MSE_KF_linear_arr.shape)
    # Standard deviation
    MSE_KF_linear_std = torch.std(MSE_KF_linear_arr)

    # Confidence interval
    KF_std_dB = 10 * torch.log10(MSE_KF_linear_std + MSE_KF_linear_avg) - MSE_KF_dB_avg

    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    print("Kalman Filter - STD:", KF_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out, KGout]


##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter")

[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out, KG_KF] = KFTest(args, test_input, test_target, allStates=Loss_On_AllState)

legend = ["Ground Truth", "MB RTS", "OBSERVE ", "KNet", "KG_KalmanFilter", "KG_KNet"]
font_size = 14
T_test = KNet_out[0].size()[1]
x_plt = range(0, T_test)
# plt.plot(x_plt, KG_KF[0][0, :].detach().cpu().numpy(), label=legend[4])
# plt.plot(x_plt, KG_KNet[0][0, :].detach().cpu().numpy(), label=legend[5])
plt.plot(x_plt, test_target[0][0, :].detach().cpu().numpy(), label=legend[0])
plt.plot(x_plt, KF_out[0][0, :], label=legend[1])
# plt.plot(x_plt, test_input[0][0, :].detach().cpu().numpy(), label=legend[2])
plt.plot(x_plt, KNet_out[0][0, :].detach().cpu().numpy(), label=legend[3])
plt.legend(fontsize=font_size)
plt.xlabel('t', fontsize=font_size)
plt.ylabel('m', fontsize=font_size)
plt.savefig("results/"+"fertilizer_weightUNet")
plt.clf()
# torch.save([test_target, KF_out, KNet_out], 'results/'+'weight-500-2000.pt')

# [test_target, KF_out, KNet_out] = torch.load('results/'+'weight-500-2000.pt', map_location=device)
# loss_fn = nn.MSELoss(reduction='mean')
# mask = torch.tensor([True, False, False, False])
# MSE_linear_LOSS_KF = loss_fn(test_target[0, mask, :], KF_out[0, mask, :])
# MSE_linear_LOSS_KNet = loss_fn(test_target[0, mask, :], KNet_out[0, mask, :])
# MSE_db_LOSS_KF = 10 * torch.log10(MSE_linear_LOSS_KF)
# MSE_db_LOSS_KNet = 10 * torch.log10(MSE_linear_LOSS_KNet)
# print("KFLOSS", MSE_db_LOSS_KF, "KNETLOSS", MSE_db_LOSS_KNet)
