import torch
from torch import nn
from torch.autograd import Variable
import math
from transformers import BertModel, BertConfig
from torch.nn.utils.rnn import pad_sequence

device = "cuda:1"
v = 0.2
dic_node = {
    "ch": 30,
    "us": 25,
    "sp": 24,
    "fr": 21
}
dic_na = {
    "ch": 141,
    "us": 268,
    "sp": 234,
    "fr": 222
}


class GRUlayer(nn.Module):
    def __init__(self, h, x, d):
        super(GRUlayer, self).__init__()
        self.linerxr = nn.Linear(x, h)
        self.linernr = nn.Linear(h, h)
        self.linerxz = nn.Linear(x, h)
        self.linernz = nn.Linear(h, h)
        self.linerxh = nn.Linear(x, h)
        self.linernh = nn.Linear(h, h)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, H, X):
        R = self.sig(self.linerxr(X) + self.linernr(H))
        Z = self.sig(self.linerxz(X) + self.linernz(H))
        H_ = self.tanh(self.linerxh(X) + self.linernh(R * H))
        return Z * H + (1 - Z) * H_


class RGATlayer(nn.Module):
    def __init__(self, d, relation, num_heads, f, node_nums, D=None):
        super(RGATlayer, self).__init__()
        if D == None:
            D = 2 * d
        self.num = node_nums
        self.f = f
        self.d = d
        self.relation = relation
        self.heads = num_heads
        w, q, K, GRU = ({}, {}, {}, {})
        self.tanh = nn.Tanh()
        for j in relation.keys():
            for k in relation[j].keys():
                GRU[str((j, k))] = GRUlayer(self.f, self.d, self.f).to(device)
                for i in range(num_heads):
                    w[str((j, k, i))] = nn.Parameter(torch.rand((d, f), requires_grad=True).to(device) * 2 * v - v)
                    q[str((j, k, i))] = nn.Parameter(torch.rand((f, D), requires_grad=True).to(device) * 2 * v - v)
                    K[str((j, k, i))] = nn.Parameter(torch.rand((f, D), requires_grad=True).to(device) * 2 * v - v)

        self.ww = nn.ParameterDict(w)
        self.qq = nn.ParameterDict(q)
        self.kk = nn.ParameterDict(K)
        self.gru = GRU


    def forward(self, h):
        h_ = torch.zeros(self.num, self.f).to(device)
        for j in self.relation.keys():
            t = torch.zeros(self.f).to(device)
            for k in self.relation[j].keys():
                df = torch.zeros(self.f).to(device)
                for i in range(self.heads):
                    H = h[self.relation[j][k]].mm(self.ww[str((j, k, i))])
                    J = h[j].reshape(1, self.d).mm(self.ww[str((j, k, i))]).reshape(1, self.f)
                    E = torch.softmax((H.mm(self.kk[str((j, k, i))]) * (J.mm(self.qq[str((j, k, i))]))).sum(axis=1),
                                      dim=0).reshape(-1, 1)  # E:(n,1) H(n,f)
                    A = (E * H).sum(axis=0)  # A:(f)
                    df += A
                df = df / self.heads
                t += self.gru[str((j, k))](df, h[j])
            t = t / len(self.relation[j].keys())
            h_[j] = self.tanh(t)
        return h_


class Graphembeding(nn.Module):
    def __init__(self, num_layers, num_nodes, d_list, relation, heads, na):
        super(Graphembeding, self).__init__()
        self.R = []
        for i in range(num_layers):
            RGAt = RGATlayer(d_list[i], relation, heads, d_list[i + 1], num_nodes)
            self.R.append(RGAt)
        self.relation = relation
        self.embeding = nn.Embedding(num_nodes, d_list[0])
        self.num = num_layers
        self.RR = nn.Sequential(*self.R)
        self.action = torch.zeros(na, d_list[-1])

    def forward(self, x):
        h = self.embeding(x)
        h = self.RR(h)
        return h


class Attention(nn.Module):
    def __init__(self, d):
        super(Attention, self).__init__()
        self.self_attention = nn.MultiheadAttention(d, 1)
        self.dense_seq = nn.Linear(9, 1)
        self.conf1=BertConfig(vocab_size=9,num_hidden_layers=2,hidden_size=d,num_attention_heads=5,
                                          hidden_dropout_prob=0.1)
        self.conf2=BertConfig(vocab_size=8,num_hidden_layers=2,hidden_size=d,num_attention_heads=5,
                                          hidden_dropout_prob=0.1)
        self.tran9=BertModel(self.conf1).to(device)
        self.tran8 = BertModel(self.conf2).to(device)
    def forward(self, X):
        X = torch.cat([X[:, :, i, :] for i in range(X.shape[2])], dim=-1).transpose(1, 0)
        X=self.tran9(inputs_embeds=X.transpose(0,1))[0].transpose(0,1)
        X = self.self_attention(X, X, X)[0].transpose(1, 0)
        return X


class CapsuleLayer(nn.Module):
    def __init__(self, in_channels, num_units, unit_size):
        super(CapsuleLayer, self).__init__()
        self.in_channels = in_channels
        self.num_units = num_units
        self.soft = nn.Softmax(dim=1)
        self.W = nn.Parameter(torch.randn((1, num_units, in_channels, unit_size, unit_size), device=device) * 2 * v - v,
                              requires_grad=True)

    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s ** 2, dim=3, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / (mag + 1e-5))
        return s

    def forward(self, x):
        batch_size = x.size(0)
        # W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = x
        # Transform inputs by weight matrix.
        u_hat = u_hat.reshape(u_hat.shape[0], u_hat.shape[2], u_hat.shape[1], u_hat.shape[3], 1)
        # Initialize routing logits to zero.
        b_ij = Variable(torch.zeros((1, self.in_channels, self.num_units, 1), device=device))
        # Iterative routing.
        num_iterations = 1
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            c_ij = self.soft(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # (batch_size, 1, num_units, unit_size, 1)
            v_j = CapsuleLayer.squash(s_j)

            # (batch_size, features, num_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update b_ij (routing)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)


class attention(nn.Module):
    def __init__(self, hid_size):
        super(attention, self).__init__()
        self.linear = nn.Linear(hid_size, hid_size, bias=False)

    def forward(self, kv, q):
        score = torch.bmm(self.linear(kv), q.unsqueeze(-1)) / math.sqrt(kv.shape[-1])
        alpha = torch.softmax(score, dim=-1)
        return torch.bmm(alpha.transpose(1, 2), kv).squeeze()


class part(nn.Module):
    def __init__(self, d, Ndevice, lens, num_heads,
                 atten_type):
        super(part, self).__init__()
        self.cur_size = -1
        self.lens = lens
        self.dense = nn.Linear(2 * d, Ndevice)
        self.dropout = nn.Dropout(0.3)
        self.index_his = -1
        self.index_cur = -1
        self.attn = attention(d)

    def forward(self, X):
        if self.cur_size != X.shape[0]:
            self.cur_size = X.shape[0]
            self.index_his = []
            for i in range(0, self.cur_size - self.lens):
                self.index_his.append(range(i, i + self.lens))
            self.index_his = torch.LongTensor(self.index_his).to(device)
            self.index_cur = (self.index_his[:, -1] + 1)
        h_his = X[self.index_his]
        h_cur = X[self.index_cur, :]
        H = self.dropout(self.attn(h_his, h_cur))
        H = torch.cat([X[:self.index_cur[0]], H], dim=0)
        h_cur = torch.cat([X[:self.index_cur[0]], h_cur.squeeze()], dim=0)
        H = H + X
        # return self.dense(torch.cat([H, h_cur], dim=-1))
        return torch.cat([H, h_cur], dim=-1)


class DeepUDI(nn.Module):
    # def __init__(self, d, vocab_lens, norm_shape_his, norm_shape_cur, num_heads, num_layers):
    def __init__(self, ed, gl, hl, vocab_lens, relation, national, flag, cluster, his_flag, cap_flag, gnn_flag):
        super(DeepUDI, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(4 * ed, vocab_lens[-1])
        # self.embedding_hour = nn.Embedding(vocab_lens[0], ed)
        # self.embedding_week = nn.Embedding(vocab_lens[1], ed)
        self.embedding_week = Time2Vec("sin", ed)
        self.embedding_hour = Time2Vec("cos", ed)
        self.embedding_d = nn.Embedding(vocab_lens[2], ed)
        self.embedding_dc = nn.Embedding(vocab_lens[3], ed)

        self.RGAT = Graphembeding(num_layers=gl, num_nodes=dic_node[national], d_list=[ed, ed, ed, ed,ed,ed,ed],
                                  relation=relation, heads=1,
                                  na=dic_na[national])
        self.flag = flag
        self.num = dic_node[national]
        self.cluster = cluster
        self.ed = ed
        self.conf1=BertConfig(vocab_size=9,num_hidden_layers=2,hidden_size=ed,num_attention_heads=5,
                                          hidden_dropout_prob=0.1)
        self.conf2=BertConfig(vocab_size=8,num_hidden_layers=2,hidden_size=ed*4,num_attention_heads=5,
                                          hidden_dropout_prob=0.1)
        self.tran9=BertModel(self.conf1).to(device)
        self.tran8 = BertModel(self.conf2).to(device)
        self.attentions = Attention(d=4 * ed)
        self.cap = CapsuleLayer(in_channels=9, num_units=8, unit_size=4 * ed)
        self.cap_dense = nn.Linear(8, 1)

        # self.part = part(200, vocab_lens[-1], lens=512, num_heads=1, atten_type="torch")
        self.part = part(4 * ed, vocab_lens[-1], lens=hl, num_heads=1, atten_type="torch")

        self.dense = nn.Linear(4 * ed, vocab_lens[-1])
        self.his_dense = nn.Linear(2 * 4 * ed, vocab_lens[-1])
        self.att_dense = nn.Linear(9, 1)

        self.his_flag = his_flag
        self.gnn_flag = gnn_flag
        self.cap_flag = cap_flag

    def forward(self, X, x_cur):
        batch_size = X.shape[0]
        X = X[:, :-1, :]

        ind = X[:, :, 2].type(torch.long)
        if self.gnn_flag:
            H = self.RGAT(torch.arange(self.num).to(device)).to(device)
            H = H[self.flag[ind].type(torch.long), :]
            X_d = H.unsqueeze(2)
        else:
            X_d = self.embedding_d(X[:, :, 2]).unsqueeze(2)

        X_h = self.embedding_hour(X[:, :, 0]).unsqueeze(2)
        X_w = self.embedding_week(X[:, :, 1]).unsqueeze(2)
        X_dc = self.embedding_dc(X[:, :, 3]).unsqueeze(2)
        X = torch.cat([X_h, X_w, X_d, X_dc], dim=2)

        X = self.attentions(X)

        # # h = torch.zeros(batch, self.k, self.vec_num, self.d * 4, 1).to(device)
        # # h = torch.zeros(512, 8, 9, 200, 1).to(device)
        if self.cap_flag:
            h = torch.zeros(batch_size, 8, 9, 4 * self.ed).to(device)
            clu = self.cluster[self.flag[ind].type(torch.long)]

            # for j in range(batch_size):
            #     for i in range(8):
            #         num = (clu[j] == i).sum()
            #         tmp = X[j, clu[j] == i]
            #         h[j, i, :num] = tmp

            for i in range(8):
                num = (clu[:] == i).sum(dim=1)
                idx = clu[:] == i

                res = torch.split(X[idx], num.cpu().numpy().tolist(), dim=0)
                res = pad_sequence(list(res))
                res = res.transpose(0, 1)

                h[:, i, :num.max()] = res

            x = self.cap(h).squeeze()
            # x = self.tran8.forward(inputs_embeds=x, position_ids=torch.zeros(x.shape[0], 8, device=device).long())[0]
            x = self.cap_dense(x.transpose(1, 2)).squeeze()
        else:
            x = self.att_dense(X.transpose(1, 2)).squeeze()

        if self.his_flag:
            x = self.part(x)
            return self.his_dense(x)
        else:
            return self.dense(x)


def t2v(tau, f, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, hiddem_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, hiddem_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, hiddem_dim)

        self.fc1 = nn.Linear(hiddem_dim, hiddem_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.fc1(x)
        return x
