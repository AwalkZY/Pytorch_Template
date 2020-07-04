import collections
import os

import torch
from torch import nn

from trainer.trainer_base import BaseTrainer
from utils.accessor import save_model, load_model
from utils.calculator import max_min_norm
from utils.helper import move_to_cuda
from utils.slimer import expand_and_repeat, union_dim, split_dim
from utils.timer import Timer


class MainTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._init()
        self._build_dataset(config.DATASET, config.TRAIN, config.TEST)
        self._build_model(config.MODEL)
        self._build_optimizer(config.OPTIMIZER)

    def _init(self):
        self.num_updates = 0

    def _build_dataset(self, dataset_config, train_config, test_config):
        import dataset as prototype
        from gensim.models import KeyedVectors
        from torch.utils.data import DataLoader
        dataset = getattr(prototype, dataset_config.NAME, None)
        vocab = KeyedVectors.load_word2vec_format(dataset_config.VOCAB_PATH, binary=True)
        self.train_set = dataset(data_path=dataset_config.TRAIN_DATA_PATH, vocab=vocab, config=dataset_config)
        # self.val_set = dataset(data_path=config.VAL_DATA_PATH, vocab=vocab, config=config)
        self.test_set = dataset(data_path=dataset_config.TEST_DATA_PATH, vocab=vocab, config=dataset_config)
        print("Train: {} samples, Test: {} samples".format(len(self.train_set), len(self.test_set)))
        self.train_loader = DataLoader(self.train_set, batch_size=train_config.BATCH_SIZE,
                                       collate_fn=self.train_set.collate_data,
                                       num_workers=train_config.NUM_WORKERS)
        self.test_loader = DataLoader(self.test_set, batch_size=test_config.BATCH_SIZE,
                                      collate_fn=self.test_set.collate_data,
                                      num_workers=test_config.NUM_WORKERS)

    def _build_model(self, config):
        import model as prototype
        device_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
        print('GPU: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        self.model = getattr(prototype, config.NAME, None)(config[config.NAME])
        self.model = self.model.cuda(device_ids[0])
        print(self.model)
        print("Number of parameter: %.2fM" % (sum([param.nelement() for param in self.model.parameters()]) / 1e6))
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.devices_ids = device_ids

    def _build_optimizer(self, config):
        from optimizers import AdamOptimizer
        from optimizers.lr_schedulers import InverseSquareRootSchedule
        parameters = list(self.model.parameters())
        args = {
            "lr": config.LR, "weight_decay": config.WEIGHT_DECAY,
            "warmup_updates": config.WARMUP_UPDATES,
            "warmup_init_lr": config.WARMUP_INIT_LR
        }
        self.optimizer = AdamOptimizer(args, parameters)
        self.lr_scheduler = InverseSquareRootSchedule(args, self.optimizer)

    def _save_model(self, checkpoint_name):
        state_dict = {
            'num_updates': self.num_updates,
            'config': self.config
        }
        save_model(self.model, checkpoint_name, state_dict)

    def _load_model(self, checkpoint_name):
        state_dict = load_model(self.model, checkpoint_name)
        self.num_updates = state_dict['num_updates']
        self.lr_scheduler.step_update(self.num_updates)

    def train(self):
        for epoch in range(1, self.config.TRAIN.MAX_EPOCH):
            print('Start Epoch: {}'.format(epoch))
            checkpoint_name = '{}-{}.pt'.format(self.config.MODEL.NAME, epoch)
            self.train_one_epoch(epoch)
            self._save_model(checkpoint_name)
            self.evaluate()
            self.evaluate(top_n=5, thresh=0.45)
            print('=' * 60)
        print('-' * 120)
        print('Done')

    def evaluate(self, top_n=1, thresh=0.0):
        pass

    def train_one_epoch(self, epoch):
        self.model.train()

        def print_log(meter):
            msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, bid, curr_lr)
            for k, v in meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            print(msg)

        display_n_batches, bid = 50, 0
        time_meter = Timer()
        for batch_idx, batch in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            net_input['props'] = net_input['props'].expand(len(self.device_ids), -1, -1)
            output = self.model(**net_input)
            loss = self.calculate_loss()

    def validate(self):
        pass

    def align_norm_loss(self, score, props, candidate_num, k, positive=True):
        batch_size, num_clips = score.size()
        prob = torch.sigmoid(score)
        idx = torch.argsort(prob, dim=-1, descending=True)
        cand_props = props[idx[:, :candidate_num]].contiguous()
        first_prop = cand_props[:, 0]
        first_prop = union_dim(expand_and_repeat(first_prop, dim=1, times=candidate_num), 0, 1)
        cand_props = union_dim(cand_props, 0, 1)
        iou = calculate_IoU_batch((first_prop[:, 0], first_prop[:, 1]), (cand_props[:, 0], cand_props[:, 1]))
        iou = split_dim(iou, 0, batch_size, candidate_num)
        align_idx = torch.argsort(iou, dim=-1, descending=True)[:, :k]
        prob_idx = idx.gather(dim=-1, index=align_idx)
        align_all_prob = prob.gather(dim=-1, index=prob_idx)
        align_prob = align_all_prob.mean(dim=-1)
        norm_prob = max_min_norm(prob, dim=-1)
        if positive:
            global_avg = norm_prob.mean()
            gap_loss = -(align_prob.softmax(dim=-1) * align_prob.log_softmax(dim=-1)).sum(dim=-1).mean()
        else:
            global_avg, gap_loss = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        return norm_prob, align_prob, global_avg, gap_loss

    def calculate_loss(self, pos_score, neg_score, props, maps, maps_mask, candidate_num, k, weights):
        """
        :param props:
        :param neg_score: (nb, depth, height, width)
        :param pos_score: (nb, depth, height, width)
        :param maps: (nb, dim, depth, height, width)
        :param maps_mask: (nb, depth, height, width)
        :param candidate_num: number of candidates
        :param k: number of final choices
        :param weights: weights of loss functions
        :return:
        """

        batch_size, hidden_size, depth, height, width = maps.size()
        norm_prob, align_pos_score, global_loss, gap_loss = self.align_norm_loss(pos_score, props, candidate_num, k,
                                                                                 positive=True)
        _, align_neg_score, _, _ = self.align_norm_loss(neg_score, props, candidate_num, k, positive=False)
        align_loss = 0
        for depth_idx in range(depth - 1):
            joint_mask = torch.logical_and(maps_mask[:, depth_idx], maps_mask[:, depth_idx + 1])
            bottom_score = maps[:, :, depth_idx].masked_select(joint_mask)
            top_score = maps[:, :, depth_idx + 1].masked_select(joint_mask)
            dist_bottom = torch.softmax(bottom_score, dim=0)
            dist_top = torch.softmax(top_score, dim=0)
            align_loss += align_loss_proto(dist_bottom, dist_top)
        final_loss = align_loss + global_loss + gap_loss
        return final_loss, {
            "align": align_loss.item(),
            "global": global_loss.item(),
            "gap": gap_loss.item()
        }


def calculate_IoU_batch(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou
