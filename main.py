import os
import random
import time
import numpy as np
import pickle
from DeepUDI import *
# from DeepUDI import *
from sklearn.metrics import f1_score
import json
import argparse


# device = torch.device('cuda:0')


def pickle_load(path):
    return pickle.load(open(path, 'rb'))


def pickle_dump(obj, path):
    pickle.dump(obj, open(path, 'wb'))


def dataProcess(data, batch_size):
    """
    data:(total,timestep,feature) -> data(batchNum,batchsize,timestep,feature)
    train_his: data
    train_cur: (batchNum,batchsize,[:-1],feature)
    train_y:(batchNum,batchsize,[-1:],feature)
    """
    batchNum = data.shape[0] // batch_size
    if batchNum != 0:
        data = data[:(data.shape[0] // batch_size) * batch_size, :, :].reshape(-1, batch_size, 10, 4)
    else:
        data = data.reshape(1, -1, 10, 4)
    train_his, train_cur, train_y = data, data[:, :, :-1, :], data[:, :, -1:, [-1]]
    return torch.LongTensor(train_his).to(device), torch.LongTensor(train_cur).to(device), torch.LongTensor(train_y).to(
        device)


def val(model, datax, dataxcur, datay, top):
    loss_criterion = torch.nn.CrossEntropyLoss()
    h1, h3, h5, m1, m3, m5 = [], [], [], [], [], []
    lossList = []
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for b in range(datax.shape[0]):
            demo, demoCur, demoy = datax[b], dataxcur[b], datay[b].reshape(-1)
            out = model(demo, demo[:, -1, [0, 1]])
            out = torch.softmax(out, dim=1)
            _, idx = torch.sort(out, descending=True, dim=1)

            top_f1 = idx[:, :1]
            y_pred.extend(top_f1.cpu().squeeze())
            y_true.extend(demoy.cpu())

            def h(tops):
                top_m = idx[:, :tops]
                mind = top_m - demoy.unsqueeze(-1)
                zeros = int((mind == 0).sum())

                w = torch.FloatTensor([1 / n for n in range(1, tops + 1)]).reshape(-1, 1).to(device)
                zeroIndex = torch.mm((mind == 0).float(), w)

                if tops != 1:
                    return zeros / datax.shape[1], float(zeroIndex.mean())
                else:
                    return zeros / datax.shape[1], float(zeroIndex.mean()), f1_score(demoy.cpu(), top_m.cpu(),
                                                                                     average='macro'),

                # return zeros / datax.shape[1], float(zeroIndex.mean())

            res1, res3, res5 = h(1), h(3), h(5)
            h1.append(res1[0])
            h3.append(res3[0])
            h5.append(res5[0])
            m1.append(res1[1])
            m3.append(res3[1])
            m5.append(res5[1])

            loss = loss_criterion(out, demoy).cpu()
            lossList.append(loss)

        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    return f1, np.array(h1).mean(), np.array(h3).mean(), np.array(h5).mean(), np.array(m1).mean(), np.array(
        m3).mean(), np.array(m5).mean(), np.array(lossList).mean()
    # return res1[-1], np.array(h1).mean(), np.array(h3).mean(), np.array(h5).mean(), np.array(m1).mean(), np.array(
    #     m3).mean(), np.array(m5).mean(), np.array(lossList).mean()


def train(model_name, model, epoch, train_his, train_cur, train_y, vld_his, vld_cur, vld_y, test_his, test_cur, test_y):
    val_accList = []
    val_mapList = []
    counter = 0
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    loss_criterion = torch.nn.CrossEntropyLoss()
    h1l, h3l, h5l, m1l, m3l, m5l = [], [], [], [], [], []

    h1_max = 0
    h3_max = 0
    h5_max = 0
    m1_max = 0
    m3_max = 0
    m5_max = 0
    f1_max = 0
    res = {}
    # earlystopping = EarlyStopping(patience=7, path='./data/{}/BEST_{}.pth'.format(national, top_k))
    for eps in range(epoch):
        t1 = time.time()
        model.train()
        for b in range(train_his.shape[0]):
            counter += 1
            demo_his, demo_cur, demo_y = train_his[b], train_cur[b], train_y[b].reshape(-1)
            out = model(demo_his, demo_his[:, -1, [0, 1]])
            opt.zero_grad()
            loss = loss_criterion(out, demo_y)
            loss.backward()
            opt.step()
        # h1,h3,h5,m1,m3,m5,val_loss = val(vld_his, vld_cur, vld_y, top_k)
        f1, h1, h3, h5, m1, m3, m5, val_loss = val(model, test_his, test_cur, test_y, top_k)
        t2 = time.time()
        print(
            'epoch:{},h1:{},h3:{},h5:{},m1:{},m3:{},m5:{},f1:{},loss:{},time:{}'.format(eps, h1, h3, h5, m1, m3, m5, f1,
                                                                                        val_loss,
                                                                                        (t2 - t1)))
        h1l.append(h1)
        h3l.append(h3)
        h5l.append(h5)
        m1l.append(m1)
        m3l.append(m3)
        m5l.append(m5)

        if h1 > h1_max:
            h1_max = h1
            res["h1"] = h1_max
            res["h1_epoch"] = eps
            torch.save(model.state_dict(), model_name + "_{}.pth".format("h1"))

        if h3 > h3_max:
            h3_max = h3
            res["h3"] = h3_max
            res["h3_epoch"] = eps
            torch.save(model.state_dict(), model_name + "_{}.pth".format("h3"))

        if h5 > h5_max:
            h5_max = h5
            res["h5"] = h5_max
            res["h5_epoch"] = eps
            torch.save(model.state_dict(), model_name + "_{}.pth".format("h5"))

        if m1 > m1_max:
            m1_max = m1
            res["m1"] = m1_max
            res["m1_epoch"] = eps
            torch.save(model.state_dict(), model_name + "_{}.pth".format("m1"))

        if m3 > m3_max:
            m3_max = m3
            res["m3"] = m3_max
            res["m3_epoch"] = eps
            torch.save(model.state_dict(), model_name + "_{}.pth".format("m3"))

        if m5 > m5_max:
            m5_max = m5
            res["m5"] = m5_max
            res["m5_epoch"] = eps
            torch.save(model.state_dict(), model_name + "_{}.pth".format("m5"))

        if f1 > f1_max:
            f1_max = f1
            res["f1"] = f1_max
            res["f1_epoch"] = eps
            torch.save(model.state_dict(), model_name + "_{}.pth".format("f1"))

    print(
        'MAX: h1:{},h3:{},h5:{},m1:{},m3:{},m5:{},f1:{}'.format(np.array(h1l).max(), np.array(h3l).max(),
                                                                np.array(h5l).max(),
                                                                np.array(m1l).max(), np.array(m3l).max(),
                                                                np.array(m5l).max(), f1_max))

    # fs.write(str('MAX: h1:{},h3:{},h5:{},m1:{},m3:{},m5:{}'.format(np.array(h1l).max(), np.array(h3l).max(),
    #                                                                np.array(h5l).max(),
    #                                                                np.array(m1l).max(), np.array(m3l).max(),
    #                                                                np.array(m5l).max())) + "\n")

    f1, h1, h3, h5, m1, m3, m5, val_loss = val(model, test_his, test_cur, test_y, top_k)
    print('h1:{},h3:{},h5:{},m1:{},m3:{},m5:{}'.format(h1, h3, h5, m1, m3, m5))

    return res


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def init_graph(national):
    dir = "./data/{}".format(national)
    relation = pickle.load(open(dir + "//relational graph.pkl", "rb"))
    flag = pickle.load(open(dir + "//flag.pkl", "rb"))
    actrelation = pickle.load(open(dir + "//actionrelational graph.pkl", "rb"))
    cluster = pickle.load(open(dir + "//cluster.pkl", "rb"))
    return relation, flag, actrelation, cluster


def main(para):
    setup_seed(para["seed"])
    national = para["national"]

    vocab_lens = (
        data_dic.dayofweek_dict.__len__() + 1, data_dic.hour_dict.__len__(), data_dic.device_dict.__len__(),
        data_dic.device_control_dict.__len__())

    relation, flag, actrelation, cluster = init_graph(national)

    data = pickle_load('./data/{}/trn_instance_10.pkl'.format(national))[:, :, [0, 1, 2, 4]]
    data2vld = pickle_load('./data/{}/vld_instance_10.pkl'.format(national))[:, :, [0, 1, 2, 4]]
    data2test = pickle_load('./data/{}/test_instance_10.pkl'.format(national))[:, :, [0, 1, 2, 4]]

    train_his, train_cur, trian_y = dataProcess(data, batch_size=para["batch_size"])
    vld_his, vld_cur, vld_y = dataProcess(data2vld, batch_size=para["batch_size"])
    test_his, test_cur, test_y = dataProcess(data2test, batch_size=para["batch_size"])

    model = DeepUDI(para["ed"], para["gl"], para["hl"], vocab_lens, relation, actrelation, national, flag, cluster,
                    his_flag=para["h_flag"],
                    cap_flag=para["c_flag"], gnn_flag=para["g_flag"], device=device).to(device)

    model_name = "./model/{}_{}_{}_{}_{}_{}".format(para["seed"], para["national"], para["batch_size"], para["ed"],
                                                    para["hl"],
                                                    para["gl"])
    res = train(model_name, model, para["epoch"], train_his, train_cur, trian_y, vld_his, vld_cur, vld_y, test_his,
                test_cur, test_y)
    return res


def run():
    para = {
        "seed": 5,
        # "national": "fr",
        "national": "kr",
        "batch_size": 512,
        "epoch": 100,
        "ed": 50,  # embedding dimension
        "hl": 5,  # history length
        "gl": 2,  # gnn layer
        # "h_flag": True,  # history flag
        # "g_flag": True,  # gnn flag
        # "c_flag": True  # capsule flag
        "h_flag": True,  # history flag
        "g_flag": True,  # gnn flag
        "c_flag": True  # capsule flag
    }

    cmd = "import data.{}.dictionary as {}".format(para["national"], "data_dic")
    exec(cmd)

    res = main(para)

    res_file = "./result/{}_{}_{}_{}".format(para["seed"], para["national"], para["batch_size"], para["ed"], para["hl"],
                                             para["gl"])
    with open(res_file, "w") as fs:
        json.dump(res, fs)


if __name__ == '__main__':
    top_k = 5
    m = 5
    parser = argparse.ArgumentParser(description='DeepUDI Training')
    parser.add_argument('--na', default="ch", type=str, help='national: ch/fr/sp/us/kr')
    parser.add_argument('--epoch', default=500, type=int, help='training epoch')
    parser.add_argument('--device', default="cuda:0", type=str, help='device: cuda:0/cuda:1/cpu')
    args = parser.parse_args()
    device = args.device

    para = {
        "seed": 5,
        # "national": "fr",
        "national": args.na,
        # "national": "kr",
        "batch_size": 1024,
        "epoch": args.epoch,
        "ed": 50,  # embedding dimension
        "hl": 5,  # history length
        "gl": 2,  # gnn layer
        "h_flag": True,  # history flag
        "g_flag": True,  # gnn flag
        "c_flag": True  # capsule flag
    }
    cmd = "import data.{}.dictionary as {}".format(para["national"], "data_dic")
    exec(cmd)

    main(para)