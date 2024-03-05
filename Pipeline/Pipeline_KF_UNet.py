import torch
import torch.nn as nn
import random
import time


class Pipeline_KF_UNet:
    def __init__(self, Time, modelName):
        super().__init__()
        self.Time = Time
        self.modelName = modelName

    def setssModel(self, m, n, T, T_test, sw):
        self.m = m
        self.n = n
        self.T = T
        self.T_test = T_test
        self.sw = sw

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, args):
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.N_steps = args.n_steps  # Number of Training Steps
        self.N_B = args.n_batch  # Number of Samples in Batch
        self.learningRate = args.lr  # Learning Rate
        self.weightDecay = args.wd  # L2 Weight Regularization - Weight Decay
        self.alpha = args.alpha  # Composition loss factor
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def NNTrain(self, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=False):
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])

        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])

        if MaskOnState:
            mask = torch.tensor([True, False, False, False])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_steps):
            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            # Training Mode
            self.model.train()
            self.model.batch_size = self.N_B

            y_training_batch = torch.zeros([self.N_B, self.m, self.T]).to(self.device)
            train_target_batch = torch.zeros([self.N_B, self.m, self.T]).to(self.device)
            x_out_training_batch = torch.zeros([self.N_B, 1, self.sw]).to(self.device)

            # Randomly select N_B training sequences
            assert self.N_B <= self.N_E  # N_B must be smaller than N_E
            n_e = random.sample(range(self.N_E), k=self.N_B)
            ii = 0
            for index in n_e:
                y_training_batch[ii, :, :] = train_input[index]
                train_target_batch[ii, :, :] = train_target[index]
                ii += 1

            train_target_batch_m = torch.zeros([self.N_B, 1, self.T]).to(self.device)
            train_target_batch_m[:, 0, :] = train_target_batch[:, 0, :]


            # Init Sequence
            self.model.InitSequence(self.T)

            # choose sw randomly   sw = 128  tt the fist location of sw in T sequence(512)
            tt = random.randint(0, 384)
            y_training_sw = torch.zeros([self.N_B, self.m, self.sw]).to(self.device)
            train_target_sw = torch.zeros([self.N_B, 1, self.sw]).to(self.device)

            for i in range(0, self.sw):
                y_training_sw[:, :, i] = y_training_batch[:, :, tt + i]
                train_target_sw[:, :, i] = train_target_batch_m[:, :, tt + i]

            # Forward Computation
            x_out_training_batch = self.model(y_training_sw)

            # Compute Training Loss
            if (MaskOnState):
                MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch[:, mask, :], train_target_sw[:, mask, :])
            else:  # no mask on state
                MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch, train_target_sw)

            # dB Loss
            self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            MSE_trainbatch_linear_LOSS.backward(retain_graph=True)  # retain_graph=True

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()
            # self.scheduler.step(self.MSE_cv_dB_epoch[ti])

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            self.model.batch_size = self.N_CV

            with torch.no_grad():

                self.T_test = cv_input.size()[-1]  # T_test is the maximum length of the CV sequences

                x_out_cv_batch = torch.empty([self.N_CV, 1, self.sw]).to(self.device)
                cv_target_m = torch.zeros([self.N_CV, 1, self.T_test]).to(self.device)
                cv_target_m[:, 0, :] = cv_target[:, 0, :]
                cv_target_sw = torch.zeros([self.N_CV, 1, self.sw]).to(self.device)

                # Init Sequence
                self.model.InitSequence(self.T_test)

                y_cv_sw = torch.zeros([self.N_CV, self.m, self.sw]).to(self.device)

                for i in range(0, self.sw):
                    y_cv_sw[:, :, i] = cv_input[:, :, tt + i]
                    cv_target_sw[:, :, i] = cv_target_m[:, :, tt + i]

                x_out_cv_batch = self.model(y_cv_sw)

                # Compute CV Loss
                MSE_cvbatch_linear_LOSS = 0

                if (MaskOnState):
                    MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch[:, mask, :], cv_target_sw[:, mask, :])
                else:
                    MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch, cv_target_sw)

                # dB Loss
                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti

                    torch.save(self.model, path_results + 'best-model.pt')

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti], "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def NNTest(self, test_input, test_target, path_results, MaskOnState=False, \
               randomInit=False, test_init=None, load_model=False, load_model_path=None, \
               test_lengthMask=None):
        # Load model
        if load_model:
            self.model = torch.load(load_model_path, map_location=self.device)
        else:
            self.model = torch.load(path_results + 'best-model.pt', map_location=self.device)

        self.N_T = test_input.shape[0]
        self.T_test = test_input.size()[-1]
        self.MSE_test_linear_arr = torch.zeros([self.N_T])

        if MaskOnState:
            mask = torch.tensor([True, False, False, False])
            if self.m == 2:
                mask = torch.tensor([True, False])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Test mode
        self.model.eval()
        self.model.batch_size = self.N_T

        torch.no_grad()
        start = time.time()

        self.model.InitSequence(self.T_test)
        x_out_test = torch.zeros([self.N_T, 1, self.T_test]).to(self.device)
        y_test_sw = torch.zeros([self.N_T, self.m, self.sw]).to(self.device)
        for t in range(0, self.T_test + 1 - self.sw):  # t the fist location of sw in T sequence
            for i in range(0, self.sw):
                y_test_sw[:, :, i] = test_input[:, :, t + i]

            x_out_test_sw = self.model(y_test_sw)
            if t == 0:
                for tt in range(0, self.sw):
                    x_out_test[:, :, tt] = x_out_test_sw[:, :, tt]
            else:
                x_out_test[:, :, self.sw + t - 1] = x_out_test_sw[:, :, self.sw - 1]

        end = time.time()
        t = end - start

        test_target_m = torch.zeros([self.N_T, 1, self.T_test]).to(self.device)
        test_target_m[:, 0, :] = test_target[:, 0, :]
        # MSE loss
        for j in range(self.N_T):  # cannot use batch due to different length and std computation
            if (MaskOnState):
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j, mask, :], test_target_m[j, mask, :]).item()
            else:
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j, :, :], test_target_m[j, :, :]).item()

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t]
