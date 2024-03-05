import torch
from datetime import datetime
from torch.distributions.multivariate_normal import MultivariateNormal
import config as config
import time
import random
import torch.nn as nn
import numpy as np
from KF_UNet.KF_UNet import KF_UNet
from Pipeline.Pipeline_KF_UNet import Pipeline_KF_UNet as Pipeline_KF_UNet
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1E4
import matplotlib.pyplot as plt

m = 4
n = 1

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

args.T = 512  # input sequence length
args.T_test = 512  # input test sequence length
### training parameters
args.use_cuda = True # use GPU or not
args.n_steps = 1000000  # number of training steps
args.n_batch = 100  #input batch size for training
args.lr = 1e-4
args.wd = 1e-4


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

#############################
###  Dataset Generation   ###
#############################

Loss_On_AllState = True  # if false: only calculate loss on position
Train_Loss_On_AllState = True  # if false: only calculate training loss on position
print("Load Original Data")
[train_target, train_input] = torch.load('results/'+'weight-train.pt', map_location=device)
[cv_target, cv_input] = torch.load('results/'+'weight-cv.pt', map_location=device)
[test_target_raw, kf3, test_input_raw] = torch.load('results/'+'weight-test-target,kf,input.pt', map_location=device)
train_input = train_input.requires_grad_(False)
cv_input = cv_input.requires_grad_(False)
test_input_raw = test_input_raw.requires_grad_(False)
test_input = test_input_raw
test_target = test_target_raw


sw = 128


print("Data Shape")
print("testset state x size:", test_target_raw.size())
print("testset observation y size:", test_input_raw.size())
print("self.N_E in pipeline", len(train_input))
print("trainset state x size:", train_target.size())
print("trainset observation y size:", train_input.size())
print("cvset state x size:", cv_target.size())
print("cvset observation y size:", cv_input.size())


# ##########################
# ### Evaluate KalmanNet ###
# ##########################
# Build Neural Network
KNet_model = KF_UNet()
KNet_model.NNBuild(m, n, args)
print("Number of trainable parameters for KNet pass 1:", sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
# Train Neural Network
KNet_Pipeline = Pipeline_KF_UNet(strTime, "KNet")
KNet_Pipeline.setssModel(m, n, args.T, args.T_test, sw)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(args)
path_results = 'CNN_UNet/'
# [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState)
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_out, RunTime] = KNet_Pipeline.NNTest(test_input, test_target, path_results, MaskOnState=not Loss_On_AllState)  # , load_model=True, load_model_path=path_results + 'best-model-512-36db.pt'


legend = ["Ground Truth", "KF", "OBSERVE ", "KNet", "KG_KalmanFilter", "KG_KNet"]
font_size = 14
T_test = test_target_raw[0].size()[1]
x_plt = range(0, T_test)
# plt.plot(x_plt, KG_KF[0][0, :].detach().cpu().numpy(), label=legend[4])
# plt.plot(x_plt, KG_KNet[0][0, :].detach().cpu().numpy(), label=legend[5])
plt.plot(x_plt, test_target_raw[0][0, :].detach().cpu().numpy(), label=legend[0])
plt.plot(x_plt, kf3[0][0, :], label=legend[1])
# plt.plot(x_plt, test_input[0][0, :].detach().cpu().numpy(), label=legend[2])
plt.plot(x_plt, KNet_out[0][0, :].detach().cpu().numpy(), label=legend[3])
plt.legend(fontsize=font_size)
plt.xlabel('t', fontsize=font_size)
plt.ylabel('m', fontsize=font_size)
plt.savefig("results/"+"fertilizer_weightCNNUNetsw=128")
plt.clf()
torch.save([test_target, kf3, KNet_out], 'results/'+'weight-CNNUNetsw1.pt')

