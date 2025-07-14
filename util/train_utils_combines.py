#!/usr/bin/python
from __future__ import print_function
import warnings
import datasets_fault

from torch.utils.data import  DataLoader
import model
from model.basenet import *
from datasets_fault.SequenceDatasets import newdataset
from util.utils import *
from tqdm import tqdm
import math
class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9 ,weight_decay=0.0005):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),lr=lr, weight_decay=weight_decay,momentum=momentum)
            self.opt_c1 = optim.SGD(self.C1.parameters(),lr=lr, weight_decay=weight_decay,momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(),lr=lr, weight_decay=weight_decay,momentum=momentum)
        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),lr=lr, weight_decay=weight_decay)
            self.opt_c1 = optim.Adam(self.C1.parameters(),lr=lr, weight_decay=weight_decay)
            self.opt_c2 = optim.Adam(self.C2.parameters(),lr=lr, weight_decay=weight_decay)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

    def setup(self):
        """
        Initialize the datasets_fault, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{args.cuda_device}")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))
        Dataset = getattr(datasets_fault, args.data_name)
        self.datasets = {}
        if isinstance(args.transfer_task[0], str):
           args.transfer_task = eval("".join(args.transfer_task))
        self.datasets['source_train'], self.datasets['target_train'], self.datasets['target_val'],self.datasets['target_noshuffle'] = Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_split(transfer_learning=True)
        self.dataloaders={}
        self.dataloaders['source_train'] = torch.utils.data.DataLoader(self.datasets['source_train'], batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,
                                                                       pin_memory=(True if self.device == 'cuda' else False),drop_last=False)
        self.dataloaders['target_val'] = torch.utils.data.DataLoader(self.datasets['target_val'],batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers,
                                                                     pin_memory=(True if self.device == 'cuda' else False),drop_last=False)
        self.dataloaders['target_noshuffle'] = torch.utils.data.DataLoader(self.datasets['target_noshuffle'],batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,
                                                                    pin_memory=(True if self.device == 'cuda' else False),drop_last=False)
        self.dataloaders['target_train'] = torch.utils.data.DataLoader(self.datasets['target_train'],
                                                                       batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,
                                                                       pin_memory=(True if self.device == 'cuda' else False), drop_last=False)
        self.G = getattr(model, args.model_name)(args.pretrained)
        self.C1 = ResClassifier(num_classes=args.class_num, num_layer=args.num_layer, num_unit=self.G.output_num(), middle=args.middle_num)
        self.C2 = ResClassifier(num_classes=args.class_num, num_layer=args.num_layer, num_unit=self.G.output_num(), middle=args.middle_num)
        self.G.cuda()
        self.C1.cuda()
        self.C2.cuda()
        self.set_optimizer(which_opt=args.opt, lr=args.lr,weight_decay=args.weightdecay)
        self.model_all = nn.Sequential(self.G, self.C1, self.C2)

    def train(self,epoch):
        """
        Training process
        :return:
        """
        args = self.args
        start = 2
        torch.cuda.manual_seed(args.seed)
        gamma = 2 / (1 + math.exp(-10 * ((epoch) / (args.max_epoch)))) - 1
        criterion = nn.CrossEntropyLoss()
        criterion_w = Weighted_CrossEntropy
        logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
        data_loader_1 = self.dataloaders['source_train']
        data_loader_2 = self.dataloaders['target_train']
        if epoch >= start:
            mem_label, all_fea, initc, all_label, all_output = obtain_label(self.dataloaders['target_noshuffle'], self.G, self.C1, self.C2,args)
            mem_label = torch.from_numpy(mem_label).cuda()
            target_datas,target_labels,unselected_data,unselected_labels = pair_selection_v1(epoch,self.dataloaders['target_noshuffle'],mem_label, args.class_num,args.knn_times,all_fea,balance_class=args.balance_class,sel_ratio=args.sel_ratio,max_epoch=args.max_epoch)
            target_datasets = newdataset(target_datas, target_labels)
            self.dataloaders['target_train'] = DataLoader(target_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=True)
            data_loader_2 = self.dataloaders['target_train']
        self.G.train()
        self.C1.train()
        self.C2.train()
        for batch_idx, data in enumerate(zip(data_loader_1, data_loader_2)):
            if epoch >= start:
                (source_inputs, source_labels, index), (target_inputs, pseudo_label_t, targets_index) = data
            else:
                (source_inputs, source_labels, index), (target_inputs, _, targets_index) = data
            source_inputs = source_inputs.float().cuda()
            target_inputs = target_inputs.float().cuda()
            inputs = torch.cat((source_inputs, target_inputs), dim=0)
            source_labels = source_labels.long().cuda()
            if epoch >=start:
                pseudo_label_t = pseudo_label_t.long().cuda()
            bs = len(source_labels)
            self.reset_grad()
            feat_s = self.G(source_inputs)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            feat_t = self.G(target_inputs)
            output_t1 = self.C1(feat_t)
            output_t2 = self.C2(feat_t)
            output_t1_s = F.softmax(output_t1, dim=1)
            output_t2_s = F.softmax(output_t2, dim=1)
            entropy_loss = Entropy(output_t1_s)
            entropy_loss += Entropy(output_t2_s)
            if epoch >= start:
                supervision_loss = criterion_w(output_t1, pseudo_label_t) + criterion_w(output_t2, pseudo_label_t)
            else:
                supervision_loss = 0
            loss1 = criterion(output_s1, source_labels)
            loss2 = criterion(output_s2, source_labels)
            all_loss1 = loss1 + loss2 + 0.1 * entropy_loss +gamma * supervision_loss
            all_loss1.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            """target domain diversity"""
            # Step B train classifier
            self.reset_grad()
            output = self.G(inputs)
            output1 = self.C1(output)
            output2 = self.C2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1_s = F.softmax(output_t1)
            output_t2_s = F.softmax(output_t2)
            loss1 = criterion(output_s1, source_labels)
            loss2 = criterion(output_s2, source_labels)
            entropy_loss = Entropy(output_t1_s)
            entropy_loss += Entropy(output_t2_s)
            loss_dis = discrepancy(output_t1, output_t2)
            all_loss = loss1 + loss2 - 1.0 * loss_dis + 0.1 * entropy_loss
            all_loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            """target domain discriminability"""
            # Step C train genrator
            for i in range(args.num_k):
                self.reset_grad()
                output = self.G(inputs)
                feat_s = output[:bs, :]
                feat_t = output[bs:, :]
                output1 = self.C1(output)
                output2 = self.C2(output)
                output_s1 = output1[:bs, :]
                output_s2 = output2[:bs, :]
                output_t1 = output1[bs:, :]
                output_t2 = output2[bs:, :]

                output_t1_s = F.softmax(output_t1)
                output_t2_s = F.softmax(output_t2)
                entropy_loss = Entropy(output_t1_s)
                entropy_loss += Entropy(output_t2_s)
                loss_dis = discrepancy(output_t1, output_t2)
                if epoch >= start:
                    index_s1 = uncertainty(args, output_s1)
                    index_s2 = uncertainty(args, output_s2)
                    index_s = index_s1 & index_s2
                    index_t1 = uncertainty(args, output_t1)
                    index_t2 = uncertainty(args, output_t2)
                    index_t = index_t1 & index_t2
                    alignment = PrototypeAlignmentWithLoss(args.class_num, args.middle_num, )
                    zloss = alignment.forward(feat_s, source_labels, index_s, feat_t, pseudo_label_t, index_t)
                else:
                    zloss = 0
                all_loss2 = loss_dis + 0.1 * entropy_loss+gamma*zloss
                all_loss2.backward()
                self.opt_g.step()

    def test(self, epoch, best_acc):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        size = 0
        print('-' * 100, '\nTesting')
        iters = iter(self.dataloaders['target_val'])
        num_iter = len(iters)
        # with torch.no_grad():
        for i in tqdm(range(num_iter), ascii=True):
            target_data, target_labels, target_index = next(iters)
            img = target_data
            label = target_labels
            img, label = img.float().cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            feat = self.G(img)
            output1 = self.C1(feat)
            output2 = self.C2(feat)
            test_loss += F.nll_loss(output1, label).data
            output_ensemble = output1 + output2
            pred1 = output1.data.max(1)[1]
            pred2 = output2.data.max(1)[1]
            pred_ensemble = output_ensemble.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            correct2 += pred2.eq(label.data).cpu().sum()
            correct3 += pred_ensemble.eq(label.data).cpu().sum()
            size += k
        test_loss = test_loss / size
        correct_add = 100. * correct3 / size
        logging.info(
            'Test set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.6f}%) Accuracy C2: {}/{} ({:.6f}%) Accuracy Ensemble: {}/{} ({:.6f}%) \n'.format(
                test_loss, correct1, size,
                100. * correct1 / size, correct2, size, 100. * correct2 / size, correct3, size,
                100. * correct3 / size))
        model_state_dic = self.model_all.state_dict()
        if (correct_add >= best_acc):
            best_acc = correct_add

            torch.save(model_state_dic,
                       os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))
        return correct_add

