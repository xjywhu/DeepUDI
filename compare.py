import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy as np
from abc import ABCMeta, abstractmethod
import time


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# CARNN
class CARNN(nn.Module):
    def __init__(self, sequenceLength=10, d_model=40, timeStamp=7 * 8, actionNum=268):
        super(CARNN, self).__init__()
        self.actionEmb = nn.Embedding(actionNum, d_model)
        self.M = clones(nn.Linear(d_model, d_model), sequenceLength)
        self.W = clones(nn.Linear(d_model, d_model), timeStamp)
        self.w = nn.Embedding(timeStamp, d_model * d_model)
        self.d = d_model
        self.sl = sequenceLength
        self.actionNum = actionNum
        self.ts = timeStamp

    def forward(self, X, t1, t2):  # t1: day of week, t2: 8 time zones
        hl = torch.zeros(X.shape[0], self.d).to(device)
        ActionEmb = self.actionEmb(X[:, :, 3])
        T = t1 * 8 + t2
        gp = torch.zeros(X.shape[0], 1).to(device)
        T0 = torch.cat((gp, T), dim=1)[:, 0:-1]
        tStamp = (T - T0 + self.ts) % self.ts
        tStamp = torch.reshape(tStamp.int(), (-1, 1))
        tTrans = self.w(tStamp)
        tTrans = torch.reshape(tTrans, (X.shape[0], -1, self.d, self.d))

        for i in range(self.sl - 1):
            t_1 = torch.squeeze(tTrans[:, i, :, :])
            t_2 = torch.unsqueeze(hl, dim=2)
            tt = torch.squeeze(torch.matmul(t_1, t_2))
            hl = torch.sigmoid(self.M[i](ActionEmb[:, i]) + tt)
        t = torch.arange(0, self.actionNum).to(device)
        deviceEmb = self.actionEmb(t)
        t_1 = torch.squeeze(tTrans[:, -1, :, :])
        t_2 = torch.unsqueeze(torch.transpose(deviceEmb, 0, 1), dim=0)
        tt = torch.squeeze(torch.matmul(t_1, t_2))
        t_1 = torch.unsqueeze(self.M[-1](hl), dim=1)
        res = torch.squeeze(torch.matmul(t_1, tt))
        return res


# Caser
class Caser(nn.Module):
    def __init__(self, args=None):
        super(Caser, self).__init__()
        self.args = args
        self.seqLen = self.args.seqLen
        self.nBatch = self.args.nBatch
        self.d_model = self.args.d_model
        self.actionNum = self.args.actionNum
        self.nh = self.args.nh
        self.nv = self.args.nv
        self.activator = nn.ReLU()
        self.actionEmb = nn.Embedding(self.actionNum, self.d_model)
        self.dropout = nn.Dropout(self.args.drop)
        # vertical conv layer
        self.convV = nn.Conv2d(1, self.nv, (self.seqLen - 1, 1))
        # horizontal conv layer
        lengths = [i + 1 for i in range(self.seqLen - 1)]
        self.convH = nn.ModuleList([nn.Conv2d(1, self.nh, (i, self.d_model)) for i in lengths])
        self.dim_v = self.nv * self.d_model
        self.dim_h = self.nh * len(lengths)
        self.linearDim = self.dim_h + self.dim_v
        # Linear Layer
        self.T = nn.Linear(self.linearDim, self.actionNum)

    def forward(self, X):
        X = X[:, :, 3]
        X = self.actionEmb(X).unsqueeze(1)
        # out, out_h, out_v = None, None, None

        out_v = self.convV(X)
        out_v = out_v.view(X.shape[0], -1)

        out_hs = list()
        for conv in self.convH:
            convOut = self.activator(conv(X).squeeze(3))
            pool_out = F.max_pool1d(convOut, convOut.size(2)).squeeze(2)
            out_hs.append(pool_out)
        out_h = torch.cat(out_hs, 1)

        # Linear
        out = torch.cat([out_v, out_h], 1)
        out = self.dropout(out)
        return self.T(out)


# FMC
class FPMC(nn.Module):
    def __init__(self, n_users, n_items, k_UI=64, k_IL=64):
        super(FPMC, self).__init__()
        self.n_users = n_users
        self.n_action = n_items
        self.k_UI = k_UI
        self.k_IL = k_IL

        self.IL = nn.Embedding(self.n_action, self.k_IL)
        self.LI = nn.Embedding(self.n_action, self.k_IL)

    def forward(self, X, tag: int):
        if tag == 1:
            X = X[:, :, 3]
            basket_prev = []
            pos_iid = []
            neg_iid = []
            for seq in X:
                for i in range(9):
                    basket_prev.append(seq[i])
                    ne = seq[i + 1]
                    pos_iid.append(ne)
                    while True:
                        iid_negative = np.random.randint(0, self.n_action)
                        if iid_negative != ne:
                            neg_iid.append(torch.tensor(iid_negative).to(device))
                            break
            basket_prev = torch.stack(basket_prev, dim=0)
            neg_iid = torch.stack(neg_iid, dim=0)
            pos_iid = torch.stack(pos_iid, dim=0)
            pos_t = self.IL(pos_iid) * self.LI(basket_prev)
            pos_x_FMC = torch.sum(pos_t, dim=1) / (self.k_IL ** (1 / 2))
            neg_t = self.IL(neg_iid) * self.LI(basket_prev)
            neg_x_FMC = torch.sum(neg_t, dim=1) / (self.k_IL ** (1 / 2))
            return pos_x_FMC, neg_x_FMC
        else:
            X = X[:, -2, 3].squeeze()
            sample = torch.tensor(range(0, self.n_action)).to(device)
            predictions = []
            for row in X:
                prev_iid = torch.full([self.n_action], row).to(device)
                pre = self.IL(sample) * self.LI(prev_iid)
                pre = torch.sum(pre, dim=1) / (self.k_IL ** (1 / 2))
                predictions.append(pre)
            predictions = torch.stack(predictions, dim=0)
            return predictions


# HMM
class _BaseHMM():
    __metaclass__ = ABCMeta

    def __init__(self, n_state=1, x_size=1, iter=20):
        self.n_state = n_state
        self.x_size = x_size
        self.start_prob = np.ones(n_state) * (1.0 / n_state)
        self.transmat_prob = np.ones((n_state, n_state)) * (1.0 / n_state)
        self.trained = False
        self.n_iter = iter

    @abstractmethod
    def _init(self, X):
        pass

    @abstractmethod
    def emit_prob(self, x):
        return np.array([0])

    @abstractmethod
    def generate_x(self, z):
        return np.array([0])

    @abstractmethod
    def emit_prob_updated(self, X, post_state):
        pass

    def generate_seq(self, seq_length):
        X = np.zeros((seq_length, self.x_size))
        Z = np.zeros(seq_length)
        Z_pre = np.random.choice(self.n_state, 1, p=self.start_prob)
        X[0] = self.generate_x(Z_pre)
        Z[0] = Z_pre

        for i in range(seq_length):
            if i == 0: continue
            # P(Zn+1)=P(Zn+1|Zn)P(Zn)
            Z_next = np.random.choice(self.n_state, 1, p=self.transmat_prob[Z_pre, :][0])
            Z_pre = Z_next
            # P(Xn+1|Zn+1)
            X[i] = self.generate_x(Z_pre)
            Z[i] = Z_pre

        return X, Z

    def X_prob(self, X, Z_seq=np.array([])):
        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))
        _, c = self.forward(X, Z)  # P(x,z)
        prob_X = np.sum(np.log(c))  # P(X)
        return prob_X

    def predict(self, X, x_next, Z_seq=np.array([]), istrain=True):
        if self.trained == False or istrain == False:
            self.train(X)

        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))
        alpha, _ = self.forward(X, Z)  # P(x,z)
        t = self.emit_prob(np.array([x_next]))
        prob_x_next = self.emit_prob(np.array([x_next])) * np.dot(alpha[X_length - 1], self.transmat_prob)

        return prob_x_next

    def decode(self, X, istrain=True):

        if self.trained == False or istrain == False:
            self.train(X)

        X_length = len(X)
        state = np.zeros(X_length)

        pre_state = np.zeros((X_length, self.n_state))
        max_pro_state = np.zeros((X_length, self.n_state))

        _, c = self.forward(X, np.ones((X_length, self.n_state)))
        max_pro_state[0] = self.emit_prob(X[0]) * self.start_prob * (1 / c[0])

        for i in range(X_length):
            if i == 0: continue
            for k in range(self.n_state):
                prob_state = self.emit_prob(X[i])[k] * self.transmat_prob[:, k] * max_pro_state[i - 1]
                max_pro_state[i][k] = np.max(prob_state) * (1 / c[i])
                pre_state[i][k] = np.argmax(prob_state)

        state[X_length - 1] = np.argmax(max_pro_state[X_length - 1, :])
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            state[i] = pre_state[i + 1][int(state[i + 1])]

        return state

    def train_batch(self, X, Z_seq=list()):
        self.trained = True
        X_num = len(X)
        self._init(self.expand_list(X))

        if Z_seq == list():
            Z = []
            for n in range(X_num):
                Z.append(list(np.ones((len(X[n]), self.n_state))))
        else:
            Z = []
            for n in range(X_num):
                Z.append(np.zeros((len(X[n]), self.n_state)))
                for i in range(len(Z[n])):
                    Z[n][i][int(Z_seq[n][i])] = 1

        for e in range(self.n_iter):
            time_st = time.time()
            print("iter: ", e)
            b_post_state = []
            b_post_adj_state = np.zeros((self.n_state, self.n_state))
            b_start_prob = np.zeros(self.n_state)
            for n in range(X_num):
                X_length = len(X[n])
                alpha, c = self.forward(X[n], Z[n])
                beta = self.backward(X[n], Z[n], c)

                post_state = alpha * beta / np.sum(alpha * beta)
                b_post_state.append(post_state)
                post_adj_state = np.zeros((self.n_state, self.n_state))
                for i in range(X_length):
                    if i == 0: continue
                    if c[i] == 0: continue
                    post_adj_state += (1 / c[i]) * np.outer(alpha[i - 1],
                                                            beta[i] * self.emit_prob(X[n][i])) * self.transmat_prob

                if np.sum(post_adj_state) != 0:
                    post_adj_state = post_adj_state / np.sum(post_adj_state)
                b_post_adj_state += post_adj_state
                b_start_prob += b_post_state[n][0]

            b_start_prob += 0.001 * np.ones(self.n_state)
            self.start_prob = b_start_prob / np.sum(b_start_prob)
            b_post_adj_state += 0.001
            for k in range(self.n_state):
                if np.sum(b_post_adj_state[k]) == 0: continue
                self.transmat_prob[k] = b_post_adj_state[k] / np.sum(b_post_adj_state[k])

            self.emit_prob_updated(self.expand_list(X), self.expand_list(b_post_state))
            print(time.time() - time_st)

    def expand_list(self, X):
        C = []
        for i in range(len(X)):
            C += list(X[i])
        return np.array(C)

    def train(self, X, Z_seq=np.array([])):
        self.trained = True
        X_length = len(X)
        self._init(X)

        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))

        for e in range(self.n_iter):
            print(e, " iter")
            alpha, c = self.forward(X, Z)  # P(x,z)
            beta = self.backward(X, Z, c)  # P(x|z)

            post_state = alpha * beta
            post_adj_state = np.zeros((self.n_state, self.n_state))
            for i in range(X_length):
                if i == 0: continue
                if c[i] == 0: continue
                post_adj_state += (1 / c[i]) * np.outer(alpha[i - 1],
                                                        beta[i] * self.emit_prob(X[i])) * self.transmat_prob

            self.start_prob = post_state[0] / np.sum(post_state[0])
            for k in range(self.n_state):
                self.transmat_prob[k] = post_adj_state[k] / np.sum(post_adj_state[k])

            self.emit_prob_updated(X, post_state)

    def forward(self, X, Z):
        X_length = len(X)
        alpha = np.zeros((X_length, self.n_state))  # P(x,z)
        alpha[0] = self.emit_prob(X[0]) * self.start_prob * Z[0]
        c = np.zeros(X_length)
        c[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / c[0]
        for i in range(X_length):
            if i == 0: continue
            alpha[i] = self.emit_prob(X[i]) * np.dot(alpha[i - 1], self.transmat_prob) * Z[i]
            c[i] = np.sum(alpha[i])
            if c[i] == 0: continue
            alpha[i] = alpha[i] / c[i]

        return alpha, c

    def backward(self, X, Z, c):
        X_length = len(X)
        beta = np.zeros((X_length, self.n_state))  # P(x|z)
        beta[X_length - 1] = np.ones((self.n_state))
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            beta[i] = np.dot(beta[i + 1] * self.emit_prob(X[i + 1]), self.transmat_prob.T) * Z[i]
            if c[i + 1] == 0: continue
            beta[i] = beta[i] / c[i + 1]

        return beta


class DiscreteHMM(_BaseHMM):
    def __init__(self, n_state=1, x_num=1, iter=20):
        _BaseHMM.__init__(self, n_state=n_state, x_size=1, iter=iter)
        self.emission_prob = np.ones((n_state, x_num)) * (1.0 / x_num)
        self.x_num = x_num

    def _init(self, X):
        self.emission_prob = np.random.random(size=(self.n_state, self.x_num))
        for k in range(self.n_state):
            self.emission_prob[k] = self.emission_prob[k] / np.sum(self.emission_prob[k])

    def emit_prob(self, x):
        prob = np.zeros(self.n_state)
        # for i in range(self.n_state): prob[i]=self.emission_prob[i][int(x[0])]
        for i in range(self.n_state): prob[i] = self.emission_prob[i][int(x)]
        return prob

    def generate_x(self, z):
        return np.random.choice(self.x_num, 1, p=self.emission_prob[z][0])

    def emit_prob_updated(self, X, post_state):
        self.emission_prob = np.zeros((self.n_state, self.x_num))
        X_length = len(X)
        for n in range(X_length):
            self.emission_prob[:, int(X[n])] += post_state[n]

        self.emission_prob += 0.1 / self.x_num
        for k in range(self.n_state):
            if np.sum(post_state[:, k]) == 0: continue
            self.emission_prob[k] = self.emission_prob[k] / np.sum(post_state[:, k])


# LSTM
class _myLSTM(nn.Module):
    def __init__(self, sequenceLength=10, hiddenSize=40, actionNum=268):
        super(_myLSTM, self).__init__()
        self.sl = sequenceLength
        self.hSize = hiddenSize
        self.actionNum = actionNum
        self.T = nn.Linear(hiddenSize, actionNum)
        self.lstm = nn.LSTM(input_size=hiddenSize, hidden_size=hiddenSize, batch_first=True)
        self.ActionEmb = nn.Embedding(actionNum, hiddenSize).to(device)

    def forward(self, X):
        X = X[:, :, 3]
        X = self.ActionEmb(X)
        h, (ht, ct) = self.lstm(X)
        ht = torch.squeeze(ht)
        return self.T(ht)


# SASRec
class SASRec(nn.Module):
    def __init__(self, d_model=50, seqlen=10, actNum=268, stack=2, mod=1):
        super(SASRec, self).__init__()
        self.seqLen = seqlen
        self.actNum = actNum
        self.mod = mod
        self.actEmb = nn.Embedding(actNum, d_model)
        self.posEmb = nn.Parameter(torch.randn(self.seqLen - 1, d_model).to(device))
        if mod == 1:
            self.d_model = d_model
            self_encoderLayer = nn.TransformerEncoderLayer(d_model, nhead=2, dim_feedforward=d_model * 4,
                                                           activation="gelu")
            self.encoder1 = nn.TransformerEncoder(self_encoderLayer, num_layers=stack)
            self.encoder2 = nn.TransformerEncoder(self_encoderLayer, num_layers=stack)
            self.T = nn.Linear(d_model, actNum)
        else:
            self.d_model = d_model
            self_encoderLayer = nn.TransformerEncoderLayer(d_model * 2, nhead=2, dim_feedforward=d_model * 8,
                                                           activation="gelu")
            self.encoders = nn.TransformerEncoder(self_encoderLayer, num_layers=stack)
            self.T = nn.Linear(d_model * 2, actNum)

    def forward(self, X):
        X = self.actEmb(X[:, :, 3])
        X = X + self.posEmb
        X = self.encoder1(X)
        X = self.encoder2(X)
        X_9 = X[:, 8, :]
        X_9 = torch.squeeze(X_9)
        return self.T(X_9)


# STAR
class STAR(nn.Module):
    def __init__(self, sequenceLength=10, d_model=40, timeStamp=7 * 8, actionNum=268):
        super(STAR, self).__init__()
        self.E = clones(nn.Linear(1, d_model).to(device), sequenceLength - 1)
        self.Q = clones(nn.Linear(d_model, d_model).to(device), sequenceLength - 1)
        self.L = clones(nn.Linear(d_model, d_model * d_model).to(device), sequenceLength - 1)
        self.M = clones(nn.Linear(d_model, d_model).to(device), sequenceLength - 1)
        self.T = nn.Linear(d_model, actionNum).to(device)
        self.d = d_model
        self.sl = sequenceLength
        self.actionNum = actionNum
        self.actionEmb = nn.Embedding(actionNum, d_model).to(device)
        self.act = nn.ReLU6()

    def forward(self, X, t1, t2):
        p0 = torch.zeros(X.shape[0], self.d)
        h0 = torch.zeros(X.shape[0], self.d)
        ActionEmb = self.actionEmb(X[:, :, 3])
        h0, p0, ActionEmb = h0.to(device), p0.to(device), ActionEmb.to(device)
        t = t1 * 8 + t2
        t = t.to(device)
        for i in range(self.sl - 1):
            p1 = self.act(self.E[i](torch.unsqueeze(t[:, i], 1).float().to(device)) + self.Q[i](p0)).to(device)
            jt = self.act(self.L[i](p1)).to(device)
            jt = torch.reshape(jt, (jt.shape[0], self.d, self.d))
            h00 = torch.unsqueeze(h0, dim=1)
            tt = torch.squeeze(torch.matmul(h00, jt))
            h1 = self.act(self.M[i](ActionEmb[:, i, :].squeeze()) + tt)
            h0 = h1
            p0 = p1
        return self.T(h0)


class deepmove(nn.Module):
    def __init__(self):
        super(deepmove, self).__init__()
        self.emdh = nn.Embedding(hourlen, d)
        self.emdw = nn.Embedding(weeklen, d)
        self.emdd = nn.Embedding(dlen, d)
        self.emddc = nn.Embedding(dclen, d)
        self.gruc = nn.GRU(4 * d, d, 1)
        self.gruh1 = nn.GRU(4 * d, d, 1)
        self.gruh2 = nn.GRU(4 * d, d, 1)
        self.dense1 = nn.Linear(9, 1)
        self.dense2 = nn.Linear(2 * d, dclen)
        self.cbatchsize = -1
        self.index_his = -1
        self.index_cur = -1
        self.hidd = 50
        self.atten = nn.MultiheadAttention(d, 1)

    def forward(self, X):
        if self.cbatchsize != X.shape[0]:
            self.cbatchsize = X.shape[0]
            self.index_his = []
            for i in range(0, self.cbatchsize - hislen):
                self.index_his.append(range(i, i + hislen))
            self.index_his = torch.LongTensor(self.index_his).to(device)
            self.index_cur = (self.index_his[:, -1] + 1)
        h1, h2, h3 = Variable(torch.zeros((1, self.cbatchsize, d))), Variable(
            torch.zeros((1, self.cbatchsize, d))), Variable(torch.zeros((1, self.cbatchsize, d)))
        wemd = self.emdw(X[:, :, 0])
        hemd = self.emdh(X[:, :, 1])
        demd = self.emdd(X[:, :, 2])
        dcemd = self.emddc(X[:, :, 3])
        X = torch.cat([wemd, hemd, demd, dcemd], dim=-1)
        y, cur_state = self.gruc(X.transpose(1, 0), h1)
        cur_state = cur_state.transpose(1, 0)
        cur_session = cur_state[self.index_cur]
        his_state, h2 = self.gruh1(X.transpose(1, 0), h2)
        his_state = self.dense1(his_state.permute(1, 2, 0)).squeeze()
        his_session = his_state[self.index_his]
        X_queryed = self.atten(cur_session.transpose(1, 0), his_session.transpose(1, 0), his_session.transpose(1, 0))[
            0].transpose(1, 0).squeeze()
        X_queryed = torch.cat([his_state[:hislen], X_queryed], dim=0)
        X_queryed = torch.cat([cur_state.squeeze(), X_queryed], dim=-1)
        return self.dense2(X_queryed)


class GNN(nn.Module):
    def __init__(self, hidden_size, step):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))
        self.in_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.out_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.f_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def cell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.in_dense(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.out_dense(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = torch.chunk(input=gi, chunks=3, dim=2)
        h_r, h_i, h_n = torch.chunk(input=gh, chunks=3, dim=2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.cell(A, hidden)
        return hidden


class SRGNN(nn.Module):
    def __init__(self):
        super(SRGNN, self).__init__()
        self.emd = nn.Embedding(dclen + 1, d)
        self.gnn = GNN(d, gnnStep)
        self.w1 = nn.Linear(d, d)
        self.w2 = nn.Linear(d, d)
        self.q = nn.Linear(d, 1, bias=False)
        self.transform = nn.Linear(2 * d, d, bias=False)
        self.init_param()

    def init_param(self):
        stdv = 1.0 / math.sqrt(d)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, items, A, rever2seq):
        hidden = self.emd(items)
        hidden = self.gnn(A.float(), hidden)
        seq = torch.stack([hidden[i][rever2seq[i]] for i in range(rever2seq.shape[0])])  # batch x time x d
        sl = seq[:, -1:, :]  # batch x 1 x d
        alpha = self.q(torch.sigmoid(self.w1(sl) + self.w2(seq)))
        sg = torch.sum(alpha * seq, dim=1)  # b,d
        sh = self.transform(torch.cat([sl.squeeze(), sg], dim=-1))
        weight = self.emd.weight  # n,d
        return torch.mm(sh, weight.transpose(1, 0))


# SmartSense
class QTE(nn.Module):
    def __init__(self, d_model, query_len, transformer_layer_num=2):
        super(QTE, self).__init__()
        self_encoderLayer = nn.TransformerEncoderLayer(d_model, nhead=2, dim_feedforward=d_model * 4)
        self.transformerEncoder = nn.TransformerEncoder(self_encoderLayer, transformer_layer_num)
        self.linear = nn.Linear(d_model, query_len)
        self.activate = nn.functional.tanh

    def forward(self, X, q):
        X = X.transpose(0, 1)
        H = self.transformerEncoder(X)
        H = H.transpose(0, 1)
        attention_wei = nn.functional.softmax(torch.matmul(self.activate(self.linear(H)), q))
        return torch.sum(torch.mul(attention_wei.unsqueeze(2), H), dim=1)


class QTEMulQc(nn.Module):
    def __init__(self, d_model, query_len, transformer_layer_num=2):
        super(QTEMulQc, self).__init__()
        self_encoderLayer = nn.TransformerEncoderLayer(d_model, nhead=2, dim_feedforward=d_model * 4)
        self.transformerEncoder = nn.TransformerEncoder(self_encoderLayer, transformer_layer_num)
        self.linear = nn.Linear(d_model, query_len)
        self.activate = nn.functional.tanh

    def forward(self, X, q):
        X = X.transpose(0, 1)
        H = self.transformerEncoder(X)
        H = H.transpose(0, 1)
        q = q.transpose(0, 1)
        temList = []
        for i in range(0, H.shape[0]):
            temList.append(nn.functional.softmax(torch.matmul(self.activate(self.linear(H[i])), q[i])))
        attention_wei = torch.stack(temList, dim=0)
        return torch.sum(torch.mul(attention_wei.unsqueeze(2), H), dim=1)


class SmartSense(nn.Module):
    def __init__(self, d_model=50, seqLen=10, targetNum=268):
        super(SmartSense, self).__init__()
        self.d_model = d_model
        self.seqLen = seqLen
        self.encoder1s = clones(QTE(d_model=d_model, query_len=d_model, transformer_layer_num=2), seqLen - 1)
        self.encoder2 = QTEMulQc(d_model=d_model, query_len=d_model * 2, transformer_layer_num=2)
        self.posEmb = nn.Parameter(torch.randn(seqLen - 1, d_model).to(device))
        self.qc = nn.Parameter(torch.randn(d_model).to(device))
        self.Emb_e1 = nn.Embedding(40, d_model)  # device
        self.Emb_e2 = nn.Embedding(targetNum, d_model)  # action
        self.Emb_z1 = nn.Embedding(7, d_model)  # day
        self.Emb_z2 = nn.Embedding(8, d_model)  # hour
        self.l1 = nn.Linear(d_model, targetNum)
        self.activate = nn.Softmax

    def forward(self, X, z1, z2):  # X [day, hour, device, action]
        dim3 = self.Emb_z1(X[:, :, 0])
        dim4 = self.Emb_z2(X[:, :, 1])
        dim1 = self.Emb_e1(X[:, :, 2])
        dim2 = self.Emb_e2(X[:, :, 3])
        X = torch.stack([dim1, dim2, dim3, dim4], 2)
        z1 = self.Emb_z1(z1[:]).squeeze()
        z2 = self.Emb_z2(z2[:]).squeeze()
        hList = []
        X = X.transpose(0, 1)
        for i in range(0, self.seqLen - 1):
            hList.append(self.encoder1s[i](X[i], self.qc))
        H = torch.stack(hList, 0).transpose(0, 1)
        z = torch.cat((z1, z2), 1).transpose(0, 1)
        S = self.encoder2(H + self.posEmb, z)
        y_hat = self.l1(S)
        return y_hat


def getRandList(start, end, discardList, m=5):
    ar = np.arange(start, end, step=1, dtype=int)
    ar = np.delete(ar, discardList.to("cpu"))
    np.random.shuffle(ar)
    return ar[:m]


def calLoss(routineDataSet, smartSense):
    loss = 0.0
    loss2 = 0.0
    for data in routineDataSet:
        data = data.to(device)
        y = torch.zeros(1, 50).to(device)
        dataEmbed = smartSense.Emb_e1(data)
        data1 = torch.cat((y, dataEmbed), dim=0)
        data2 = torch.cat((dataEmbed, y), dim=0)
        diag = torch.log(torch.sigmoid(torch.diag(torch.matmul(data2, data1.transpose(0, 1)))))
        loss2 += torch.sum(diag) - diag[0] * 2
        pRi = getRandList(0, 39, data)
        for i in range(0, data.shape[0]):
            for j in range(0, len(pRi)):
                loss2 += torch.log(
                    torch.sigmoid(-torch.matmul(dataEmbed[i], smartSense.Emb_e1(torch.tensor(pRi[j]).to(device)))))
    loss -= loss2
    return loss
