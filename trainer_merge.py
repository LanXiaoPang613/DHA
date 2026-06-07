import os
import numpy as np
import torch as t
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter, writer

CHECKPOINTS_DIR = "./checkpoints"

import torch
import torch.nn.functional as F
import copy

def loss_coteaching(y1, y2, targets, forget_rate: float=0.):
    """
    y1, y2: [B, C] logits
    targets: [B] long
    """
    device = y1.device
    # reduction='none' 才能逐样本筛
    l1 = F.cross_entropy(y1, targets, reduction='none')  # [B] on device
    l2 = F.cross_entropy(y2, targets, reduction='none')  # [B] on device

    ind1 = torch.argsort(l1)  # small loss first
    ind2 = torch.argsort(l2)

    remember_rate = 1.0 - forget_rate
    num_remember = max(1, int(remember_rate * targets.size(0)))

    ind1_update = ind1[:num_remember]
    ind2_update = ind2[:num_remember]

    # exchange
    loss1_update = F.cross_entropy(y1[ind2_update], targets[ind2_update], reduction='mean')
    loss2_update = F.cross_entropy(y2[ind1_update], targets[ind1_update], reduction='mean')
    return loss1_update, loss2_update

class FocalLoss(t.nn.Module):
    def __init__(self, alpha=1., gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = t.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = t.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

def mix_data_lab(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    lam = max(lam, 1 - lam)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, index, lam

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
            # 只考虑iid noise的话，每个client训练样本数一样，所以不用做nk/n
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg


def mixup_criterion(pred, y_a, y_b, lam):
    c = F.log_softmax(pred, 1)
    return lam * F.cross_entropy(c, y_a) + (1 - lam) * F.cross_entropy(c, y_b)


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained
                 model2,
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 optim2=None,
                 scheduler=None,               # Schedule for learning rate decay
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True):                   # Whether to use the GPU
                 
        self._model = model
        self._model2 = model2
        self._crit = crit
        self._optim = optim
        self._optim2 = optim2
        self._scheduler = scheduler # 2个调度器
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda   # bool类型
        self._focal_loss = FocalLoss()
        # self.penlty=NegEntropy()
        self.rate_schedule = np.ones(400) * 0.
        # self.rate_schedule[:20] = np.linspace(0, 0.1 ** 1, 20)

        if cuda:
            self._model = model.cuda()
            self._model2 = model2.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        if not os.path.exists(CHECKPOINTS_DIR):
            os.mkdir(CHECKPOINTS_DIR)
        t.save({'state_dict': self._model.state_dict()}, CHECKPOINTS_DIR + '/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load(CHECKPOINTS_DIR + '/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m2 = self._model2.cpu()
        m.eval()
        m2.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        y2 = self._model2(x)
        t.onnx.export(m,                 # Model being run
              x,                         # Model input (or a tuple for multiple inputs)
              fn,                        # Where to save the model (can be a file or file-like object)
              export_params=True,        # Store the trained parameter weights inside the model file
              opset_version=11,          # The ONNX version to export the model to
              do_constant_folding=True,  # Whether to execute constant folding for optimization
              input_names = ['input'],   # The model's input names
              output_names = ['output'], # The model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # Variable lenght axes
                            'output' : {0 : 'batch_size'}})
        t.onnx.export(m2,  # Model being run
                      x,  # Model input (or a tuple for multiple inputs)
                      fn,  # Where to save the model (can be a file or file-like object)
                      export_params=True,  # Store the trained parameter weights inside the model file
                      opset_version=11,  # The ONNX version to export the model to
                      do_constant_folding=True,  # Whether to execute constant folding for optimization
                      input_names=['input'],  # The model's input names
                      output_names=['output'],  # The model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # Variable lenght axes
                                    'output': {0: 'batch_size'}})
        
    
    def val_test_step(self, x, y):
        # Propagate through the network and calculate loss
        views = [
            x,
            # t.flip(x, dims=[3]),  # H flip (W 维)
            # t.flip(x, dims=[2]),  # V flip (H 维)
            # t.flip(t.flip(x, [2]), [3]),  # HV flip
        ]

        prob_sum = 0.0
        for xv in views:
            logits1 = self._model(xv)
            logits2 = self._model2(xv)

            p1 = F.softmax(logits1, dim=1)
            p2 = F.softmax(logits2, dim=1)

            # 两网络融合（概率平均比 logits 平均更稳）
            prob_sum = prob_sum + 0.5 * (p1 + p2)

        probs = prob_sum / len(views)
        # loss = F.nll_loss(t.log(probs + 1e-12), y, reduction="mean")
        return 0., probs
        # y_pred = self._model(x)
        # y_pred2 = self._model2(x)
        # y_pred = (y_pred+y_pred2)/2.
        # l = self._crit(y_pred, y)
        # return l, y_pred

    def train_epoch(self, epoch,alpha):
        self._train_dl.mode = "train"
        self._model.train()
        self._model2.train()

        total_loss, total_loss2 = 0.0, 0.0

        for batch in tqdm(self._train_dl, desc="Training"):
            # 你 train 集会返回 4 个：x,y,types,prob（见 data.py）:contentReference[oaicite:7]{index=7}
            x, y, types, probs = batch
            x, y = (x.cuda(non_blocking=True), y.cuda(non_blocking=True)) if self._cuda else (x, y)

            self._optim.zero_grad(set_to_none=True)
            self._optim2.zero_grad(set_to_none=True)

            y1 = self._model(x)


            if epoch < 5:
                y2 = self._model2(x)
                # warmup：先各自用全量 CE
                loss1 = F.cross_entropy(y1, y, reduction='mean')
                loss2 = F.cross_entropy(y2, y, reduction='mean')
            else:
                # mix_a1, y_a1, y_b1, _, lam1 = mix_data_lab(x, y)
                # y1 = self._model(mix_a1)
                # loss1 = mixup_criterion(y1, y_a1, y_b1, lam1)
                loss1 = F.cross_entropy(y1, y, reduction='mean')
                mix_a, y_a, y_b, _, lam= mix_data_lab(x, y,alpha=alpha)
                y2 = self._model2(mix_a)
                loss2 = mixup_criterion(y2, y_a, y_b, lam)

            self._optim.zero_grad()
            loss1.backward()
            self._optim.step()
            self._optim2.zero_grad()
            loss2.backward()
            self._optim2.step()

            total_loss += float(loss1.detach().item())
            total_loss2 += float(loss2.detach().item())

        for s in self._scheduler:
            s.step()

        avg_loss = total_loss / len(self._train_dl)
        avg_loss2 = total_loss2 / len(self._train_dl)
        print("TRAIN loss:", avg_loss, avg_loss2)
        return avg_loss, avg_loss2
        
    def train_epoch2(self, epoch):
        self._train_dl.mode = "train"
        self._model.train()
        self._model2.train()

        total_loss = 0
        total_loss2 = 0

        # Iterate through the training set and compute total loss
        for x, y, types, probs in tqdm(self._train_dl, desc="Training"):
            x, y = (x.cuda(), y.cuda()) if self._cuda else (x, y)
            self._optim.zero_grad()
            self._optim2.zero_grad()
            # Propagate through the network and calculate loss
            y_pred = self._model(x)
            y_pred2 = self._model2(x)

            # co-teaching 方法减少0.3333这种图片
            loss_1, loss_2 = loss_coteaching(y_pred, y_pred2, y, forget_rate=self.rate_schedule[epoch])
            # 原始方法
            # loss = self._crit(y_pred, y)
            # f_loss = self._focal_loss(y_pred, y)
            # loss = f_loss
            # loss.backward()
            # self._optim.step()
            # total_loss += loss.item()

            # Compute gradient by backward propagation and update weights
            self._optim.zero_grad()
            loss_1.backward()
            self._optim.step()
            self._optim2.zero_grad()
            loss_2.backward()
            self._optim2.step()
            total_loss += loss_1.item()
            total_loss2 += loss_2.item()

        
        for s in self._scheduler:
            s.step()
        
        # Calculate the average loss for the epoch and return it
        avg_loss = total_loss / len(self._train_dl)
        avg_loss2 = total_loss2 / len(self._train_dl)
        print("TRAIN loss: ", avg_loss, avg_loss2)

        return avg_loss, avg_loss2
    
    
    def calculate_metrics(self, total_loss, y_preds, y_grounds):
        avg_loss = total_loss / len(self._val_test_dl)

        # f1_crack = f1_score(y_true=y_grounds[:, 0, 0], y_pred=y_preds[:, 0, 0], average='binary')
        # f1_inactive = f1_score(y_true=y_grounds[:, 0, 1], y_pred=y_preds[:, 0, 1], average='binary')
        # f1_mean = (f1_crack + f1_inactive) / 2
        # y_grounds = [ele for ele in y_grounds]
        y_grounds,y_preds = np.array(y_grounds), np.array(y_preds)
        f1_mean = f1_score(y_grounds, y_preds, average='weighted')
        f1_mean_de = f1_score(y_grounds, y_preds,  pos_label=1, zero_division=0)
        acc = accuracy_score(y_grounds, y_preds)

        balanced_acc = balanced_accuracy_score(y_grounds, y_preds)
        precision = precision_score(y_grounds, y_preds, average='weighted', zero_division=0)
        precision_de = precision_score(y_grounds, y_preds, pos_label=1, zero_division=0)
        recall = recall_score(y_grounds, y_preds, average='weighted', zero_division=0)
        recall_de = recall_score(y_grounds, y_preds, pos_label=1, zero_division=0)


        # print("#Cracks: ", len(y_preds[y_preds[:, 0, 0] == 1]))
        # print("#Inactive: ", len(y_preds[y_preds[:, 0, 1] == 1]))
        # print("#Both: ", len(y_preds[(y_preds[:, 0, 0] == 1) & (y_preds[:, 0, 1] == 1)]))
        print('********************************\n')
        print("VAL loss: ", avg_loss)
        print('confusion matrix:', confusion_matrix(y_grounds, y_preds))
        # print("\nF1 crack: ", f1_crack)
        # print("F1 inactive: ", f1_inactive)
        print("F1 mean: ", f1_mean,f1_mean_de, "\n")
        print("ACC: ", acc, "\n")
        print("BALANCE ACC: ", balanced_acc, "\n")
        print("PRECISION: ", precision,precision_de, "\n")
        print("RECALL: ", recall,recall_de, "\n")
        print('##############################\n')

        return avg_loss, f1_mean, acc, balanced_acc, precision, recall


    @t.no_grad()
    def val_test(self):
        self._val_test_dl.mode = "val"
        self._model.eval()
        self._model2.eval()
        
        y_preds = []
        y_grounds = []

        total_loss = 0

        # Iterate through the validation set
        for x, y in tqdm(self._val_test_dl,desc="Validation"):
            x, y = (x.cuda(), y.cuda()) if self._cuda else (x, y)
            l, y_pred = self.val_test_step(x, y)
            # total_loss += l.item()

            # Save the predictions and the labels for each batch
            # output_label1 = y_pred.ge(0.5).int()
            output_label = t.argmax(y_pred, dim=1).cpu()
            # output_label = t.nn.functional.one_hot(output_label, num_classes=len(t.unique(y))).numpy()
            y_preds.extend(output_label)
            y_grounds.extend(y.cpu().numpy())
        

        # Calculate relevant metrics and return them
        return self.calculate_metrics(total_loss, y_preds, y_grounds)

    
    def fit(self, epochs=-1,args=None):
        assert epochs > 0
        global CHECKPOINTS_DIR
        CHECKPOINTS_DIR = CHECKPOINTS_DIR+"/"+str(args.runpath)
        # Save average train and validation and f1-score for each epoch
        train_losses = []
        train_losses_2 = []
        val_losses = []
        val_f1 = []

        epoch = 0
        writer = SummaryWriter(log_dir=CHECKPOINTS_DIR)
        if args is not None:
            merge_t = args.merge_t
        else:
            merge_t = 5

        while True:
            
            if epoch == epochs:
                break

            print("--- Epoch", epoch, "---")
            train_lo, train_lo2 = self.train_epoch(epoch,args.alpha)

            if merge_t > 0:
                if epoch % merge_t ==0 and epoch > merge_t:
                    local_weights=[]
                    local_weights.append(self._model.state_dict())
                    local_weights.append(self._model2.state_dict())
                    w_glob = FedAvg(local_weights)
                    self._model.load_state_dict(w_glob)
                    self._model2.load_state_dict(w_glob)

            train_losses.append(train_lo)
            train_losses_2.append(train_lo2)
            avg_loss, f1_mean,acc,balanced_accuracy, precision, recall = self.val_test()
            writer.add_scalar('Loss/train', train_lo, epoch)
            writer.add_scalar('Loss/train2', train_lo2, epoch)
            # writer.add_scalar('Loss/val', avg_loss, epoch)
            writer.add_scalar('F1/val', f1_mean, epoch)
            writer.add_scalar('Acc/val', acc, epoch)
            writer.add_scalar('Balanced_Acc/val', balanced_accuracy, epoch)
            writer.add_scalar('precision/val', precision, epoch)
            writer.add_scalar('recall/val', recall, epoch)

            val_losses.append(avg_loss)
            val_f1.append(f1_mean)

            # Save model if it reaches a certain f1-score
            # if f1_mean > 0.6:
            #     self.save_checkpoint(epoch)

            epoch += 1

        writer.close()

        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(CHECKPOINTS_DIR)
        ea.Reload()
        tags = ea.Tags()['scalars']
        with open(CHECKPOINTS_DIR+'/output.txt', 'w') as f:
            for tag in tags:
                if tag == 'Acc/val' or tag == 'Balanced_Acc/val':
                    events = ea.Scalars(tag)
                    for e in events:
                        f.write(f"{tag}\t{e.step}\t{e.value}\n")
        return train_losses, train_losses_2, val_losses, val_f1

#  bash tensorboard --logdir=./checkpoints