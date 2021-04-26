import torch
import pandas as pd
import numpy as np
import model

# test_item是测试集中item的列表，test_attribute是test_item对应的属性
test_item = pd.read_csv('data/test/test_item.csv', header=None).loc[:]
test_item = np.array(test_item)
test_attribute = pd.read_csv('data/test/test_attribute.csv', header=None).loc[:]
test_attribute = np.array(test_attribute)
user_attribute_matrix = pd.read_csv('data/test/user_attribute.csv', header=None)
user_attribute_matrix = torch.tensor(np.array(user_attribute_matrix[:]), dtype=torch.float)
ui_matrix = pd.read_csv('data/test/ui_matrix.csv', header=None)
ui_matrix = np.array(ui_matrix[:])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_test_data():
    return torch.tensor(test_item), torch.tensor(test_attribute, dtype=torch.long)


def get_similar_user(item_user, k):
    item_user = item_user.to(device)
    user_embed_matrix = user_attribute_matrix.to(device)
    similar_matrix = torch.matmul(item_user, user_embed_matrix.T)
    # 取内积最大的几个user
    index = torch.argsort(-similar_matrix)
    torch.set_printoptions(profile="full")
    return index[:, 0:k]


def dcg_k(r, k):
    r = np.asfarray(r)
    val = np.sum((np.power(2, r)-1)/(np.log2(np.arange(2, r.size+2))))
    return val


def ndcg_k(item, item_user, k):
    k_similar_user = get_similar_user(item_user, k)
    sum = 0.0
    for test_i, test_user_list in zip(item, k_similar_user):
        r = []
        for user in test_user_list:
            r.append(ui_matrix[user][test_i])
        r_ideal = sorted(r, reverse=True)
        ideal_dcg = dcg_k(r_ideal, k)
        if ideal_dcg == 0:
            sum += 0
        else:
            sum += (dcg_k(r, k)/ideal_dcg)
    return sum/item.__len__()


def p_at_k(item, item_user, k):
    k_similar_user = get_similar_user(item_user, k)
    count = 0
    test_batch_size = item.__len__()
    for test_i, test_user_list in zip(item, k_similar_user):
        for test_u in test_user_list:
            if ui_matrix[test_u, test_i] == 1:
                count += 1
    # 计算预测的k个用户，有几个用户与item有过交互，除以k
    p_k = count / (test_batch_size * k)
    return p_k


def to_valuate(item, item_user):
    similar_user = get_similar_user(item_user, 10)
    # print("similar_user:", similar_user)
    p10 = p_at_k(item, item_user, 10)
    p20 = p_at_k(item, item_user, 20)
    ndcg_10 = ndcg_k(item, item_user, 10)
    ndcg_20 = ndcg_k(item, item_user, 20)
    print("          p_10:%.4f, p_20:%.4f, ndcg_10:%.4f,ndcg_20:%.4f" % (p10, p20, ndcg_10, ndcg_20))
    columns = [p10, p20, ndcg_10, ndcg_20]
    df = pd.DataFrame(columns=columns)
    df.to_csv('data/result/test_result.csv', line_terminator='\n', index=False, mode='a', encoding='utf8')


def load_model_to_test(model_path):
    g = model.Generator()
    g.load_state_dict(torch.load(model_path))
    item, attr = get_test_data()
    g.to(device)
    item = item.to(device)
    attr = attr.to(device)
    item_user = g(attr)
    to_valuate(item, item_user)