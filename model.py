import torch
from torch import nn
import time
import torch.utils.data
from tqdm import tqdm
import load_data
import test


# 超参数设置
alpha = 0                           # 正则项参数
user_emb_dim = attr_num = 18        # item、user属性数量
attr_present_dim = 5                # 属性维度
batch_size = 1024                   # batchsize
hidden_dim = 100                    # 隐藏层维度
learning_rate = 0.0001              # 学习率
device = torch.device('cuda')       # 使用gpu训练
epoch = 200                         # 迭代次数


def is_instance(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.xavier_normal_(m.bias.unsqueeze(0))

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.G_attr_matrix = nn.Embedding(2*attr_num, attr_present_dim)
        self.l1 = nn.Linear(attr_num*attr_present_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        self.h = nn.Tanh()
        for i in self.G_attr_matrix.modules():
            nn.init.xavier_normal_(i.weight)                # 生成高斯分布
        for i in self.modules():
            is_instance(i)

    def forward(self, attribute_id):
        attr_present = self.G_attr_matrix(attribute_id)
        attr_feature = torch.reshape(attr_present, [-1, attr_num*attr_present_dim])
        out1 = self.h(self.l1(attr_feature))
        out2 = self.h(self.l2(out1))
        out3 = self.h(self.l3(out2))
        return out3

# 判别器：
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_attr_matrix = nn.Embedding(2*attr_num, attr_present_dim)
        self.l1 = nn.Linear(attr_num*attr_present_dim+user_emb_dim, hidden_dim, bias=True)
        self.h = nn.Tanh()
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        for i in self.D_attr_matrix.modules():
            torch.nn.init.xavier_normal_(i.weight)
        for i in self.modules():
            is_instance(i)

    def forward(self, attribute_id, user_emb):
        attribute_id = attribute_id.long()
        attr_present = self.D_attr_matrix(attribute_id)
        attr_feature = torch.reshape(attr_present, [-1, attr_num*attr_present_dim])
        emb = torch.cat((attr_feature, user_emb), 1)
        emb = emb.float()
        out1 = self.h(self.l1(emb))
        out2 = self.h(self.l2(out1))
        d_logit = self.l3(out2)
        d_prob = torch.sigmoid(d_logit)             # 调用sigmoid函数
        return d_prob, d_logit


def train(g, d, train_loader, neg_loader, epochs, g_optim, d_optim, datasetLen):
    g = g.to(device)
    d = d.to(device)
    print("++++++++++++++++ Training on cuda +++++++++++++++")
    loss = nn.BCELoss()               # 二分类的交叉熵损失函数
    start = time.time()

    for epo in tqdm(range(epochs)):
        i = 0
        neg_iter = neg_loader.__iter__()
        # 训练D
        d_loss_sum = 0.0
        for user, item, attr, user_emb in train_loader:
            if i*batch_size >= datasetLen:
                break
            # 取出负采样的样本
            _, _, neg_attr, neg_user_emb = neg_iter.next()
            neg_attr = neg_attr.to(device)
            neg_user_emb = neg_user_emb.to(device)
            attr = attr.to(device)
            user_emb = user_emb.to(device)
            fake_user_emb = g(attr)                         # 根据item的属性生成用户表达
            d_real, d_logit_real = d(attr, user_emb)
            d_fake, d_logit_fake = d(attr, fake_user_emb)
            d_neg, d_logit_neg = d(neg_attr, neg_user_emb)
            # d_loss分成三部分, 正样本，生成的样本，负样本
            d_optim.zero_grad()
            d_loss_real = loss(d_real, torch.ones_like(d_real))
            d_loss_fake = loss(d_fake, torch.zeros_like(d_fake))
            d_loss_neg = loss(d_neg, torch.zeros_like(d_neg))
            d_loss_sum = torch.mean(d_loss_real + d_loss_fake+d_loss_neg)
            d_loss_sum.backward()
            d_optim.step()
            i += 1
        # 训练G
        g_loss = 0.0
        for user, item, attr, user_emb in train_loader:
            # g loss
            g_optim.zero_grad()
            attr = attr.long()
            attr = attr.to(device)
            fake_user_emb = g(attr)
            fake_user_emb.to(device)
            d_fake, _ = d(attr, fake_user_emb)
            g_loss = loss(d_fake, torch.ones_like(d_fake))
            g_loss.backward()
            g_optim.step()
        end = time.time()
        print("\nepoch:",epo+1," time:%.2fs, d_loss:%.6f, g_loss:%.6f " % ((end - start), d_loss_sum, g_loss))
        start = end
        # test---
        item, attr = test.get_test_data()
        item = item.to(device)
        attr = attr.to(device)
        item_user = g(attr)
        test.to_valuate(item, item_user)
        g_optim.zero_grad()                         # 梯度清零

        # 保存模型
        print("++++++++++++++++ model has been saved! ++++++++++++++++")
        torch.save(g.state_dict(), 'data/result/g_'+str(epo)+".pt")
        torch.save(d.state_dict(), 'data/result/d_' + str(epo) + ".pt")


def run():
    train_dataset = load_data.LoadDataset('data/train/train_data.csv', 'data/train/user_emb.csv')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    neg_dataset = load_data.LoadDataset('data/train/neg_data.csv', 'data/train/user_emb.csv')
    neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    generator = Generator()
    discriminator = Discriminator()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=alpha)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=alpha)
    # 因为负样本的数据量要小一些，为了训练方便，使用负样本的长度来训练
    train(generator, discriminator, train_loader, neg_loader, epoch, g_optimizer, d_optimizer, neg_dataset.__len__())
