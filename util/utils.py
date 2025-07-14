import os
from datetime import datetime
from torch.utils.data.dataset import ConcatDataset
import logging
import importlib
from torch import optim
import torch.nn as nn
from scipy.spatial.distance import cdist
import numpy as np
import torch
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_accuracy(preds, targets):
    assert preds.shape[0] == targets.shape[0]
    # correct = torch.eq(preds.argmax(dim=1), targets).float().sum().item()
    correct = torch.eq(preds, targets).float().sum().item()
    accuracy = correct / preds.shape[0]
    return accuracy
def get_next_batch(loaders, iters, src, device, return_idx=False):
    inputs, labels = None, None
    if type(src) == list:
        for key in src:
            try:
                inputs, labels, src_idx = next(iters[key])
                break
            except StopIteration:
                continue
        if inputs == None:
            for key in src:
                iters[key] = iter(loaders[key])
            inputs, labels, src_idx = next(iters[src[0]])
    else:
        try:
            inputs, labels, src_idx = next(iters[src])
        except StopIteration:
            iters[src] = iter(loaders[src])
            inputs, labels, src_idx = next(iters[src])

    if return_idx:
        return inputs.to(device), labels.to(device), src_idx.to(device)
    else:
        return inputs.to(device), labels.to(device)
def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    if logger.root.hasHandlers():
        for i in logger.root.handlers:
            logger.root.removeHandler(i)
    return logger
def creat_file(args):
    # prepare the saving path for the model
    source = ''
    for src in args.source_name:
        source += src
    file_name = '[' + source + ']' + 'To' + '[' + \
                args.target + ']' + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_path = os.path.join(save_dir, file_name)

    # set the logger
    logger = setlogger(args.save_path + '.log')

    # save the args
    for k, v in args.__dict__.items():
        if k != 'source_name':
            logging.info("{}: {}".format(k, v))
    return logger, args

def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1)- F.softmax(out2)))

def discrepancy_matrix(out1, out2):
    out1 = F.softmax(out1,dim=1)
    out2 = F.softmax(out2,dim=1)
    mul = out1.transpose(0, 1).mm(out2)
    loss_dis = torch.sum(mul) - torch.trace(mul)
    return loss_dis

def loss_11(output_t1,output_t2):
    mul = output_t1 * output_t2
    cdd_loss1 = torch.sqrt(torch.sum(mul))
    return cdd_loss1

def loss_12(output_t1,output_t2):
    mul1 = torch.sum(output_t1 * output_t1)
    mul2 = torch.sum(output_t2 * output_t2)
    cdd_loss2 = torch.sqrt(mul1+mul2)
    return cdd_loss2

def EntropyLoss(input_):
    mask = input_.ge(0.000001)###与0.000001对比，大于则取1，反之取0
    mask_out = torch.masked_select(input_, mask)##平铺成为一维向量
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))##计算熵
    return entropy / float(input_.size(0))

def Entropy_div(input_):
    epsilon = 1e-5
    input_ = torch.mean(input_, 0) + epsilon
    entropy = input_ * torch.log(input_)
    entropy = torch.sum(entropy)
    return entropy

def Entropy_condition(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1).mean()
    return entropy

def Entropy(input_):
    return Entropy_condition(input_) + Entropy_div(input_)

def Weighted_CrossEntropy(input_,labels):
    input_s = F.softmax(input_)
    entropy = -input_s * torch.log(input_s + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    weight = 1.0 + torch.exp(-entropy)
    weight = weight / torch.sum(weight).detach().item()
    #print("cross:",nn.CrossEntropyLoss(reduction='none')(input_, labels))
    return torch.mean(weight * nn.CrossEntropyLoss(reduction='none')(input_, labels))



def CombinedWeightedCrossEntropy(input_, labels, class_weights):
    # 固定的类别权重损失
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')(input_, labels)

    # 基于熵的样本权重
    input_s = F.softmax(input_, dim=1)
    entropy = -input_s * torch.log(input_s + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    sample_weights = 1.0 + torch.exp(-entropy)
    sample_weights = sample_weights / torch.sum(sample_weights).detach().item()

    # 结合两者的权重
    weighted_ce_loss = sample_weights * ce_loss
    return torch.mean(weighted_ce_loss)
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
def select_supports(ent, labels, num_classes, supports, scores):
    ent_s = ent
    y_hat = labels.argmax(dim=1).long()
    # y_hat = labels
    filter_K = 100
    if filter_K == -1:
        indices = torch.LongTensor(list(range(len(ent_s))))
    else:
        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        for i in range(num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_K])

        indices = torch.cat(indices)

    supports = supports[indices]
    labels = labels[indices]
    ent = ent[indices]
    scores = scores[indices]

    return supports, labels


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits.float(), dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    return kl_div
def prototype_loss(z,p,labels=None,use_hard=False,tau=1):
    #z [batch_size,feature_dim]
    #p [num_class,feature_dim]
    #labels [batch_size,]
    z = F.normalize(z,1)
    p = F.normalize(p,1)
    dist = z @ p.T / tau
    # if labels is None:
    _,labels = dist.max(1)
    labels= F.one_hot(labels)
    if use_hard:
        """use hard label for supervision """
        #_,labels = dist.max(1)  #for prototype-based pseudo-label
        labels = labels.argmax(1)  #for logits-based pseudo-label
        loss =  F.cross_entropy(dist,labels)
    else:
        """use soft label for supervision """
        loss = softmax_kl_loss(labels.detach(),dist).sum(1).mean(0)  #detach is **necessary**
        #loss = softmax_kl_loss(dist,labels.detach()).sum(1).mean(0) achieves comparable results
    return dist,loss


def topk_cluster(feature, supports, scores, p, k=3):
    # p: outputs of model batch x num_class
    feature = F.normalize(feature, 1)
    supports = F.normalize(supports, 1)
    sim_matrix = feature @ supports.T  # B,M
    topk_sim_matrix, idx_near = torch.topk(sim_matrix, k, dim=1)  # batch x K
    scores_near = scores[idx_near].detach().clone()  # batch x K x num_class
    diff_scores = torch.sum((p.unsqueeze(1) - scores_near) ** 2, -1)

    loss = -1.0 * topk_sim_matrix * diff_scores
    return loss.mean()

def obtain_label(loader, netE, netC1, netC2, args, c=None):
    start_test = True
    netE.eval()
    netC1.eval()
    netC2.eval()
    predict = np.array([], dtype=np.int64)
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            # indexs = data[2]
            inputs = inputs.float().cuda()
            # labels= labels.cuda()
            feas = netE(inputs)
            outputs1 = netC1(feas)
            outputs2 = netC2(feas)
            outputs = outputs1 + outputs2
            #torch.stack([outputs1,outputs2]).mean(dim=0)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    #print("all_label:",all_label.size()[0],"right:",torch.squeeze(predict).float().eq(all_label.data).sum().item())
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Only source accuracy = {:.2f}% -> After the clustering = {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str+'\n')

    return pred_label.astype('int'), torch.from_numpy(all_fea).cuda(), initc, all_label.float().cuda(), all_output.cuda()

def pair_selection_v1(epoch, test_loader, labels, class_num, knn_times, train_features, balance_class=True,
                   sel_ratio=0,max_epoch=300,corrected=False,):
    '''
    k_val:  neighbors number of knn
    labels: pseudo-labels obtained from feature prototypes
    '''
    similarity_graph_all = torch.zeros(len(test_loader.dataset), len(test_loader.dataset))  # 初始化相似度图矩阵
    labels = labels.long()  # 将伪标签转换为长整型
    train_labels = labels.clone().cuda()  # 复制伪标签到train_labels，并将其放置在GPU上
    discrepancy_measure = torch.zeros((len(test_loader.dataset),)).cuda()  # 初始化不一致度测量向量
    agreement_measure = torch.zeros((len(test_loader.dataset),))  # 初始化一致度测量向量

    with torch.no_grad():
        for i in range(knn_times):
            print(f'starting the {i+1}st knn....')

            retrieval_one_hot_train = torch.zeros(5, class_num).cuda()
            train_new_labels = train_labels.clone()

            for batch_idx, (_, _, index) in enumerate(test_loader):
                batch_size = index.size(0)
                features = train_features[index]

                # 计算相似度图
                dist = torch.mm(features, train_features.t())
                similarity_graph_all[index] = dist.cpu().detach()

                dist[torch.arange(dist.size(0)), index] = -1

                # 选择k个最近邻
                yd, yi = dist.topk(5, dim=1, largest=True, sorted=True)
                candidates = train_new_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot_train.resize_(batch_size * 5, class_num).zero_()
                retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)

                yd_transform = torch.exp(yd.clone().div_(5))
                probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batch_size, -1, class_num),
                                                      yd_transform.view(batch_size, -1, 1)), 1)

                probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

                prob_temp = probs_norm[torch.arange(0, batch_size), labels[index]]
                prob_temp[prob_temp <= 1e-4] = 1e-4
                prob_temp[prob_temp > (1 - 1e-4)] = 1 - 1e-4
                discrepancy_measure[index] = -torch.log(prob_temp)

                # 更新标签
                sorted_pro, predictions_corrected = probs_norm.sort(1, True)
                new_labels = predictions_corrected[:, 0]
                train_labels[index] = new_labels
                agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1] == labels[index]).float().data.cpu()

            selected_examples = agreement_measure  #

    if balance_class:
        agreement_measure = torch.zeros((len(labels),)).cuda()
        sel_ratio = sel_ratio + (1.0-sel_ratio)*epoch/max_epoch
        for i in range(class_num):
            idx_class = labels == i
            num_per_class = idx_class.sum()
            idx_class = (idx_class.float() == 1.0).float().nonzero().squeeze()
            discrepancy_class = discrepancy_measure[idx_class]
            k_corrected = sel_ratio * num_per_class
            if k_corrected >= 1:
                top_clean_class_relative_idx = \
                    torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=True)[1]

                i_sel_index = idx_class[top_clean_class_relative_idx]
                agreement_measure[i_sel_index] = 1.0
                if corrected:
                    the_val = discrepancy_class[top_clean_class_relative_idx[-1]]
                    the_val_index = (discrepancy_class == the_val).float().nonzero().squeeze().long()
                    agreement_measure[idx_class[the_val_index]] = 1.0
        selected_examples = agreement_measure
    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()
        target_datas = []
        target_labels = []
        selected_indices = list(set(index_selected))
        for i in range(len(selected_indices)):
            target_datas.append(test_loader.dataset.seq_data[selected_indices[i]])
            target_labels.append(labels[selected_indices[i]])

        target_datas = torch.tensor(target_datas)

        target_labels = torch.tensor(target_labels)
        target_datas = np.array(target_datas.cpu())
        target_labels = np.array(target_labels.cpu())
        right_score=0
        for i in range(len(index_selected)):
            # testY true label
            if test_loader.dataset.labels[selected_indices[i]] == labels[selected_indices[i]]:
                right_score += 1
        clean_accuracy = right_score / len(index_selected)
        logstr = f'selection samples accuracy:{100 * clean_accuracy:.2f}%'
        print(logstr)
        all_indices = set(range(len(test_loader.dataset)))
        selected_indices = set(index_selected.tolist())
        unselected_indices = list(all_indices - selected_indices)
        unselected_data=[]
        unselected_labels=[]
        for idx in range(len(unselected_indices)):
            unselected_data.append(test_loader.dataset.seq_data[unselected_indices[idx]])
            unselected_labels.append(labels[unselected_indices[idx]])
        unselected_data = torch.tensor(unselected_data)

        unselected_labels = torch.tensor(unselected_labels)
        unselected_data = np.array(unselected_data.cpu())
        unselected_labels = np.array(unselected_labels.cpu())

        return target_datas,target_labels,unselected_data,unselected_labels


def one_hot_embedding(labels, num_classes=13):
    # Convert to One Hot Encoding
    device = labels.device  # Get the device of labels
    y = torch.eye(num_classes, device=device)
    return y[labels]

def uncertainty_prob(args,preds):
    alpha = relu_evidence(preds) + 1  # B,C
    u_s1 = args.class_num / torch.sum(alpha, dim=1, keepdim=True)  # B,1
    return u_s1


def uncertainty(args,preds):
    alpha = relu_evidence(preds) + 1  # B,C
    u_s1 = args.class_num / torch.sum(alpha, dim=1, keepdim=True)
    value, index = torch.kthvalue(u_s1.view(-1, ), round(args.batch_size * 0.85))
    index = (u_s1 < value)

    return index


def relu_evidence(y):
    # return y
    # return F.relu(y)
    return torch.exp(torch.clamp(y, -10, 10))


def kl_divergence(alpha, num_class, device):
    uni = torch.ones([1, num_class], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(uni).sum(dim=1, keepdim=True)
            - torch.lgamma(uni.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - uni)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl_loss = first_term + second_term
    return kl_loss


def evidential_loss(func, target, alpha, epoch_num, num_classes, annealing_step, device=None):
    target = target.to(device)
    # alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)  # B,1
    A = torch.sum(target * (func(S) - func(alpha)), dim=1, keepdim=True)  # B,C
    # 正则化
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - target) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    # print(A)
    # print("KL",kl_div)
    return A + kl_div * 0.005


def evidential_criterion(output, target, epoch_num, num_classes, annealing_step, device=None):
    evidence = relu_evidence(output)  # B,T
    alpha = evidence + 1
    loss = evidential_loss(torch.digamma,
                           target,  # B,C
                           alpha, epoch_num,
                           num_classes, annealing_step, device)  # torch.mean

    return loss

def probality_estimate(output):
    evidence = relu_evidence(output)  # B,C
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)  # B,1
    return alpha / S
def UncertaintyLoss(std):
    return torch.norm(std, dim=1).mean(0)



class PrototypeAlignmentWithLoss:
    def __init__(self, num_classes, feature_dim, gamma=0.9, tau=5, device='cuda'):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Initialize dynamic dictionary for source prototypes
        self.dynamic_prototypes_s = torch.zeros((num_classes, feature_dim)).to(self.device)
        self.dynamic_prototypes_t = torch.zeros((num_classes, feature_dim)).to(self.device)

    def compute_class_prototypes(self, features, pseudo_labels):
        class_prototypes = torch.zeros((self.num_classes, self.feature_dim)).to(self.device)
        for k in range(self.num_classes):
            mask = (pseudo_labels == k)
            selected_features = features[mask]
            if selected_features.shape[0] > 0:
                class_prototypes[k] = selected_features.mean(dim=0)
        return class_prototypes

    def target_prototypes(self, target_features, target_pseudo_labels, index_t):
        class_prototypes = torch.zeros((self.num_classes, self.feature_dim)).to(self.device)
        index_t = index_t.squeeze()
        for k in range(self.num_classes):
            mask = (target_pseudo_labels == k)
            mask_t = index_t & mask
            selected_features = target_features[mask_t]
            if selected_features.shape[0] > 0:
                class_prototypes[k] = selected_features.mean(dim=0)
        return class_prototypes

    def cosine_similarity(self, a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))

    def compute_omega(self, a, b):
        return torch.exp(self.cosine_similarity(a, b) / self.tau)

    def class_domain_loss(self, source_prototypes, target_prototypes):
        loss = 0
        for k in range(self.num_classes):
            squared_difference = torch.square(source_prototypes[k] - target_prototypes[k])
            mean_squared_difference = torch.mean(squared_difference)
            loss += mean_squared_difference
        return loss

    def class_prototype_alignment_loss(self, features, prototypes, pseudo_labels):
        loss = 0
        for k in range(self.num_classes):
            mask = (pseudo_labels == k)
            selected_features = features[mask]
            if selected_features.shape[0] > 0:
                prototype = prototypes[k]
                distances = torch.norm(selected_features - prototype, dim=1)
                loss += torch.mean(distances)
        return loss/self.num_classes

    def forward(self, source_features, source_logits, index_s, target_features, target_logits, index_t):
        source_pseudo_labels = source_logits
        target_pseudo_labels = target_logits

        current_source_prototypes = self.compute_class_prototypes(source_features, source_pseudo_labels)
        dynamic_prototypes_s = self.gamma * current_source_prototypes + (1 - self.gamma) * self.dynamic_prototypes_s

        current_target_prototypes = self.compute_class_prototypes(target_features, target_pseudo_labels)
        dynamic_prototypes_t = self.gamma * current_target_prototypes + (1 - self.gamma) * self.dynamic_prototypes_t

        inter_domain_loss = self.class_domain_loss(dynamic_prototypes_s, dynamic_prototypes_t)
        source_intra_domain_loss = self.class_prototype_alignment_loss(source_features, dynamic_prototypes_s, source_pseudo_labels)
        target_intra_domain_loss = self.class_prototype_alignment_loss(target_features, dynamic_prototypes_t, target_pseudo_labels)

        total_loss = inter_domain_loss + source_intra_domain_loss + target_intra_domain_loss
        return total_loss
