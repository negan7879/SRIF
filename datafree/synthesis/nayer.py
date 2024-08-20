import copy

import datafree
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook, InstanceMeanHook
from datafree.criterions import jsdiv, get_image_prior_losses, kldiv
from datafree.utils import ImagePool, DataIter, clip_images
from torchvision import transforms
from kornia import augmentation
import time
import numpy as np
import math

def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67)  # , alpha=40


def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)  # , alpha=0.67


def reset_l0(model):
    for n, m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)

def reset_g(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.Linear)):
            nn.init.normal_(m.weight, mean=0, std=1)
            nn.init.constant_(m.bias, 0)


def reset_g1(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, (nn.Linear)):
            nn.init.normal_(m.weight, mean=0, std=1)
            nn.init.constant_(m.bias, 0)


def reset_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


def custom_cross_entropy(preds, target):
    return torch.mean(torch.sum(-target * preds.log_softmax(dim=-1), dim=-1))

def calculate_normalized_mcd(pred, T_label, num_classes):
    def calculate_gmm_centers(pred, T_label, num_classes):
        class_centers = torch.zeros((num_classes, pred.shape[1]), dtype=pred.dtype, device=pred.device)
        for k in range(num_classes):
            class_samples = pred[T_label == k]
            if class_samples.shape[0] > 0:
                class_centers[k] = class_samples.mean(dim=0)
        return class_centers

    def calculate_mcd(pred, T_label, class_centers):
        num_samples = pred.shape[0]
        mcd = torch.zeros(num_samples, dtype=pred.dtype, device=pred.device)
        for i in range(num_samples):
            distances = torch.norm(pred[i] - class_centers, dim=1)
            class_distance = distances[T_label[i]]
            other_distances = torch.cat([distances[:T_label[i]], distances[T_label[i] + 1:]])
            min_confusion_distance = other_distances.min() / class_distance
            mcd[i] = min_confusion_distance
        return mcd

    def normalize_mcd(mcd):
        max_mcd = mcd.max()
        return mcd / max_mcd if max_mcd != 0 else mcd

    # 计算GMM类中心
    class_centers = calculate_gmm_centers(pred, T_label, num_classes)

    # 计算MCD
    mcd = calculate_mcd(pred, T_label, class_centers)

    # 归一化MCD
    normalized_mcd = normalize_mcd(mcd)

    return normalized_mcd


class NAYER(BaseSynthesis):
    def __init__(self, teacher, student, generator, num_classes, img_size,
                 init_dataset=None, g_steps=100, lr_g=0.1,
                 synthesis_batch_size=128, sample_batch_size=128,
                 adv=0.0, bn=1, oh=1,
                 save_dir='run/fast', transform=None, autocast=None, use_fp16=False,
                 normalizer=None, device='cpu', distributed=False,
                 warmup=10, bn_mmt=0, bnt=30, oht=1.5,
                 cr_loop=1, g_life=50, g_loops=1, gwp_loops=10, dataset="cifar10"):
        super(NAYER, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size
        self.g_steps = g_steps

        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.bn_mmt = bn_mmt

        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = int(synthesis_batch_size/cr_loop)
        self.sample_batch_size = sample_batch_size
        self.init_dataset = init_dataset
        self.use_fp16 = use_fp16
        self.autocast = autocast  # for FP16
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None
        self.generator = generator.to(device).train()
        self.device = device
        self.hooks = []

        self.ep = 0
        self.ep_start = warmup

        self.g_life = g_life
        self.bnt = bnt
        self.oht = oht
        self.g_loops = g_loops
        self.gwp_loops = gwp_loops
        self.dataset = dataset
        self.label_list = torch.LongTensor([i for i in range(self.num_classes)])

        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))

        if dataset == "imagenet" or dataset == "tiny_imagenet":
            self.aug = transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                normalizer,
            ])
        else:
            self.aug = transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ])

    def jitter_and_flip(self, inputs_jit, lim=1. / 8., do_flip=True):
        lim_0, lim_1 = int(inputs_jit.shape[-2] * lim), int(inputs_jit.shape[-1] * lim)

        # apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

        # Flipping
        flip = random.random() > 0.5
        if flip and do_flip:
            inputs_jit = torch.flip(inputs_jit, dims=(3,))
        return inputs_jit



    def synthesize(self, targets=None):
        start = time.time()
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        best_oh = 1e6

        if (self.ep - self.ep_start) % self.g_life == 0 or self.ep % self.g_life == 0:
            self.generator = self.generator.reinit()

        if self.ep < self.ep_start:
            g_loops = self.gwp_loops
        else:
            g_loops = self.g_loops
        self.ep += 1
        bi_list = []
        if g_loops == 0:
            return None, 0, 0, 0
        if self.dataset == "imagenet":
            idx = torch.randperm(self.label_list.shape[0])
            self.label_list = self.label_list[idx]
        for gs in range(g_loops):
            best_inputs = None
            best_s_feature = None
            s_feature = None
            best_s_output = None
            self.generator.re_init_le()

            if self.dataset == "imagenet":
                targets, ys = self.generate_ys_in(cr=0.0, i=gs)
                print(targets)
            else:
                targets, ys = self.generate_ys(cr=0.0)
            ys = ys.to(self.device)
            targets = targets.to(self.device)

            optimizer = torch.optim.Adam([
                {'params': self.generator.parameters()},
            ], lr=self.lr_g, betas=[0.5, 0.999])

            for it in range(self.g_steps):
                inputs = self.generator(targets=targets)
                if self.dataset == "imagenet":
                    inputs = self.jitter_and_flip(inputs)
                    inputs_aug = self.aug(inputs)
                else:
                    inputs_aug = self.aug(inputs)

                # t_out = self.teacher(inputs_aug)
                t_out, t_feature = self.teacher(inputs_aug, True)
                loss_bn = sum([h.r_feature for h in self.hooks])
                loss_oh = custom_cross_entropy(t_out, ys.detach())

                if self.adv > 0 and (self.ep > self.ep_start):
                    # s_out, s_feature = self.student(inputs_aug, True)
                    s_out = self.student(inputs_aug, False)
                    # s_feature = s_feature.detach()
                    mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                    loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(
                        1) * mask).mean()  # decision adversarial distillation
                else:
                    loss_adv = loss_oh.new_zeros(1)

                loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv

                if loss_oh.item() < best_oh:
                    best_oh = loss_oh

                # print("%s - bn %s - bn %s - oh %s - adv %s" % (
                # it, (loss_bn * self.bn).data, loss_bn.data, (loss_oh).data, (self.adv * loss_adv).data))

                with torch.no_grad():
                    if best_cost > loss.item() or best_inputs is None:
                        best_cost = loss.item()
                        best_inputs = inputs.data
                        if self.ep > self.ep_start:
                            best_s_feature = t_feature.detach()
                            best_s_output = t_out.detach()





                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.bn_mmt != 0:
                for h in self.hooks:
                    h.update_mmt()



            # if (best_s_output is not  None) and (best_s_feature is not  None) and (self.ep > (self.ep_start + 80)):
            if (best_s_output is not  None) and (best_s_feature is not  None) and False:
                # print("save")
                best_s_output = nn.Softmax(dim=1)(best_s_output)
                best_s_feature = (best_s_feature.t() / torch.norm(best_s_feature, p=2, dim=1)).t()
                best_s_feature = best_s_feature.float()
                uniform = torch.ones(len(best_s_feature), self.num_classes) / self.num_classes
                uniform = uniform.cuda()

                pi = best_s_output.sum(dim=0)
                mu = torch.matmul(best_s_output.t(), (best_s_feature))
                mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

                zz, gamma = self.gmm((best_s_feature), pi, mu, uniform)
                # pred_label = gamma.argmax(dim=1)

                for round in range(1):
                    pi = gamma.sum(dim=0)
                    mu = torch.matmul(gamma.t(), (best_s_feature))
                    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

                    zz, gamma = self.gmm((best_s_feature), pi, mu, gamma)
                    pred_label = gamma.argmax(axis=1)

                LPG = calculate_normalized_mcd(zz, torch.argmax(best_s_output, dim=1), self.num_classes )

                # sort_zz = zz.sort(dim=1, descending=True)[0]
                # zz_sub = sort_zz[:, 0] - sort_zz[:, 1]
                #
                #
                # LPG = zz_sub / zz_sub.max()
                # LPG = nn.Softmax(dim=0)(zz_sub)

                PPL = best_s_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()

                JMDS = (LPG * PPL)
                # JMDS = PPL

                # 1. 根据 JMDS 排序
                sorted_indices = torch.argsort(JMDS, descending=True)  # 从高到低排序的索引

                # 2. 选择前 70%
                num_to_keep = int(0.9 * len(JMDS))
                selected_indices = sorted_indices[:num_to_keep]

                # 3. 选择前 70% 的 best_inputs
                best_inputs = best_inputs[selected_indices]


                # print("hello")
            self.student.train()
            end = time.time()


            self.data_pool.add(best_inputs)


            bi_list.append(best_inputs)

            dst = self.data_pool.get_dataset(transform=self.transform)
            if self.init_dataset is not None:
                init_dst = datafree.utils.UnlabeledImageDataset(self.init_dataset, transform=self.transform)
                dst = torch.utils.data.ConcatDataset([dst, init_dst])
            if self.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
            else:
                train_sampler = None
            loader = torch.utils.data.DataLoader(
                dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
                num_workers=4, pin_memory=True, sampler=train_sampler)
            self.data_iter = DataIter(loader)
        return {"synthetic": bi_list}, end - start, best_cost, best_oh

    def sample(self):
        return self.data_iter.next()

    def gmm(self, all_fea, pi, mu, all_output):
        Cov = []
        dist = []
        log_probs = []
        epsilon = 1e-6
        for i in range(len(mu)):
            temp = all_fea - mu[i]
            predi = all_output[:, i].unsqueeze(dim=-1)
            Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / (predi.sum()) + epsilon * torch.eye(
                temp.shape[1]).cuda()
            try:
                chol = torch.linalg.cholesky(Covi)
            except RuntimeError:
                Covi += epsilon * torch.eye(temp.shape[1]).cuda() * 100
                chol = torch.linalg.cholesky(Covi)
            chol_inv = torch.inverse(chol)
            Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
            logdet = torch.logdet(Covi)
            mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
            log_prob = -0.5 * (Covi.shape[0] * np.log(2 * math.pi) + logdet + mah_dist) + torch.log(pi)[i]
            Cov.append(Covi)
            log_probs.append(log_prob)
            dist.append(mah_dist)
        # Cov = torch.stack(Cov, dim=0)
        # dist = torch.stack(dist, dim=0).t()
        log_probs = torch.stack(log_probs, dim=0).t()
        zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
        gamma = torch.exp(zz)

        return zz, gamma

    def generate_ys_in(self, cr=0.0, i=0):
        target = self.label_list[i*self.synthesis_batch_size:(i+1)*self.synthesis_batch_size]
        target = torch.tensor([250, 230, 283, 282, 726, 895, 554, 555, 105, 107])


        ys = torch.zeros(self.synthesis_batch_size, self.num_classes)
        ys.fill_(cr / (self.num_classes - 1))
        ys.scatter_(1, target.data.unsqueeze(1), (1 - cr))

        return target, ys

    def generate_ys(self, cr=0.0):
        s = self.synthesis_batch_size // self.num_classes
        v = self.synthesis_batch_size % self.num_classes
        target = torch.randint(self.num_classes, (v,))
        for i in range(s):
            tmp_label = torch.tensor(range(0, self.num_classes))
            target = torch.cat((tmp_label, target))

        ys = torch.zeros(self.synthesis_batch_size, self.num_classes)
        ys.fill_(cr / (self.num_classes - 1))
        ys.scatter_(1, target.data.unsqueeze(1), (1 - cr))
        # print(target)

        return target, ys

    def generate_lys(self, cr=0.0, value=3):
        s = self.synthesis_batch_size // self.num_classes
        v = self.synthesis_batch_size % self.num_classes
        target = torch.randint(self.num_classes, (v,))
        for i in range(s):
            tmp_label = torch.tensor(range(0, self.num_classes))
            target = torch.cat((tmp_label, target))

        yf = torch.zeros(self.synthesis_batch_size, self.num_classes)
        yf.scatter_(1, target.data.unsqueeze(1), (1 - cr))
        yf = yf.to(device=self.device)

        yl = torch.ones(self.synthesis_batch_size, self.num_classes)*(-value)
        yl.scatter_(1, target.data.unsqueeze(1), value)
        yl = yl.to(device=self.device)

        cr_vec = torch.ones(size=(self.synthesis_batch_size, self.num_classes), device=self.device)*cr

        return target, yf, yl, cr_vec


    def generate_lys_v2(self, cr=0.0, value=3, norm=50):
        s = self.synthesis_batch_size // self.num_classes
        v = self.synthesis_batch_size % self.num_classes
        target = torch.randint(self.num_classes, (v,))
        crate = random.randint(0, int(cr*norm))/norm
        for i in range(s):
            tmp_label = torch.tensor(range(0, self.num_classes))
            target = torch.cat((tmp_label, target))

        yf = torch.zeros(self.synthesis_batch_size, self.num_classes)
        yf.scatter_(1, target.data.unsqueeze(1), (1 - crate))
        yf = yf.to(device=self.device)

        yl = torch.ones(self.synthesis_batch_size, self.num_classes)*(-value)
        yl.scatter_(1, target.data.unsqueeze(1), value)
        yl = yl.to(device=self.device)

        cr_vec = torch.ones(size=(self.synthesis_batch_size, self.num_classes), device=self.device)*crate

        return target, yf, yl, cr_vec


    def generate_lys_v3(self, cr=0.0):
        s = self.synthesis_batch_size // self.num_classes
        v = self.synthesis_batch_size % self.num_classes
        target = torch.randint(self.num_classes, (v,))
        for i in range(s):
            tmp_label = torch.tensor(range(0, self.num_classes))
            target = torch.cat((tmp_label, target))

        yf = torch.zeros(self.synthesis_batch_size, self.num_classes)
        yf.scatter_(1, target.data.unsqueeze(1), (1 - cr))

        yf = yf.to(device=self.device)

        yl = torch.zeros(size=(self.synthesis_batch_size, self.num_classes), device=self.device)
        cr_vec = torch.ones(size=(self.synthesis_batch_size, self.num_classes), device=self.device)*cr

        return target, yf, yl, cr_vec