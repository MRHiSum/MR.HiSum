# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from networks.mlp import SimpleMLP
from networks.pgl_sum.pgl_sum import PGL_SUM
from networks.vasnet.vasnet import VASNet
from networks.sl_module.sl_module import SL_module

from model.utils.evaluation_metrics import evaluate_summary
from model.utils.generate_summary import generate_summary
from model.utils.evaluate_map import generate_mrhisum_seg_scores, top50_summary, top15_summary


class Solver(object):
    def __init__(self, config=None, train_loader=None, val_loader=None, test_loader=None):
        
        self.model, self.optimizer, self.writer, self.scheduler = None, None, None, None

        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.global_step = 0

        self.criterion = nn.MSELoss(reduction='none').to(self.config.device)

    def build(self):
        """ Define your own summarization model here """
        # Model creation
        if self.config.model == 'MLP':
            self.model = SimpleMLP(1024, [1024], 1)
            self.model.to(self.config.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)

        elif self.config.model == 'PGL_SUM':
            self.model = PGL_SUM(input_size=1024, output_size=1024, num_segments=4, heads=8, fusion="add", pos_enc="absolute")
            self.model.to(self.config.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.97)
            self.init_weights(self.model, init_type='xavier')

        elif self.config.model == 'VASNet':
            self.model = VASNet(hidden_dim=1024)
            self.model.to(self.config.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
            self.init_weights(self.model, init_type='xavier')

        elif self.config.model == 'SL_module':
            self.model = SL_module(input_dim=1024, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5)
            self.model.to(self.config.device)
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.l2_reg)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

        else:
            print("Wrong model")
            exit()
    
    def train(self):
        best_f1score = -1.0
        best_map50 = -1.0
        best_map15 = -1.0
        best_f1score_epoch = 0
        best_map50_epoch = 0
        best_map15_epoch = 0
        
        for epoch_i in range(self.config.epochs):
            print("[Epoch: {0:6}]".format(str(epoch_i)+"/"+str(self.config.epochs)))
            self.model.train()

            loss_history = []
            num_batches = int(len(self.train_loader))
            iterator = iter(self.train_loader)

            for _ in tqdm(range(num_batches)):

                self.optimizer.zero_grad()
                data = next(iterator)

                frame_features = data['features'].to(self.config.device)
                gtscore = data['gtscore'].to(self.config.device)
                mask = data['mask'].to(self.config.device)

                score, weights = self.model(frame_features, mask)
                loss = self.criterion(score[mask], gtscore[mask]).mean()

                loss.backward()
                loss_history.append(loss.item())
                
                self.optimizer.step()

            loss = np.mean(np.array(loss_history))
            
            val_f1score, val_map50, val_map15 = self.evaluate(dataloader=self.val_loader)
            
            if best_f1score <= val_f1score:
                best_f1score = val_f1score
                best_f1score_epoch = epoch_i
                f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, f'best_f1.pkl')
                torch.save(self.model.state_dict(), f1_save_ckpt_path)

            if best_map50 <= val_map50:
                best_map50 = val_map50
                best_map50_epoch = epoch_i
                map50_save_ckpt_path = os.path.join(self.config.best_map50_save_dir, f'best_map50.pkl')
                torch.save(self.model.state_dict(), map50_save_ckpt_path)
            
            if best_map15 <= val_map15:
                best_map15 = val_map15
                best_map15_epoch = epoch_i
                map15_save_ckpt_path = os.path.join(self.config.best_map15_save_dir, f'best_map15.pkl')
                torch.save(self.model.state_dict(), map15_save_ckpt_path)
            
            print("   [Epoch {0}] Train loss: {1:.05f}".format(epoch_i, loss))
            print('    VAL  F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5}'.format(val_f1score, val_map50, val_map15))
            
        print('   Best Val F1 score {0:0.5} @ epoch{1}'.format(best_f1score, best_f1score_epoch))
        print('   Best Val MAP-50   {0:0.5} @ epoch{1}'.format(best_map50, best_map50_epoch))
        print('   Best Val MAP-15   {0:0.5} @ epoch{1}'.format(best_map15, best_map15_epoch))

        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write('   Best Val F1 score {0:0.5} @ epoch{1}\n'.format(best_f1score, best_f1score_epoch))
        f.write('   Best Val MAP-50   {0:0.5} @ epoch{1}\n'.format(best_map50, best_map50_epoch))
        f.write('   Best Val MAP-15   {0:0.5} @ epoch{1}\n\n'.format(best_map15, best_map15_epoch))
        f.flush()
        f.close()

        return f1_save_ckpt_path, map50_save_ckpt_path, map15_save_ckpt_path

    def evaluate(self, dataloader=None):
        """ Saves the frame's importance scores for the test videos in json format.

        :param int epoch_i: The current training epoch.
        """
        self.model.eval()
        
        fscore_history = []
        map50_history = []
        map15_history = []

        dataloader = iter(dataloader)
        
        for data in dataloader:
            frame_features = data['features'].to(self.config.device)
            gtscore = data['gtscore'].to(self.config.device)

            if len(frame_features.shape) == 2:
                seq = seq.unsqueeze(0)
            if len(gtscore.shape) == 1:
                gtscore = gtscore.unsqueeze(0)

            B = frame_features.shape[0]
            mask=None
            if 'mask' in data:
                mask = data['mask'].to(self.config.device)

            with torch.no_grad():
                score, attn_weights = self.model(frame_features, mask=mask)

            # Summarization metric
            score = score.squeeze().cpu()
            gt_summary = data['gt_summary'][0]
            cps = data['change_points'][0]
            n_frames = data['n_frames']
            nfps = data['n_frame_per_seg'][0].tolist()
            picks = data['picks'][0].numpy()
            
            machine_summary = generate_summary(score, cps, n_frames, nfps, picks)
            # print("MACHINE", machine_summary, machine_summary.shape)
            # print("GT SUMMARY",gt_summary, gt_summary.shape)
            f_score, kTau, sRho = evaluate_summary(machine_summary, gt_summary, eval_method='avg')
            fscore_history.append(f_score)

            # Highlight Detection Metric
            gt_seg_score = generate_mrhisum_seg_scores(gtscore.squeeze(0), uniform_clip=5)
            gt_top50_summary = top50_summary(gt_seg_score)
            gt_top15_summary = top15_summary(gt_seg_score)
            
            highlight_seg_machine_score = generate_mrhisum_seg_scores(score, uniform_clip=5)
            highlight_seg_machine_score = torch.exp(highlight_seg_machine_score) / (torch.exp(highlight_seg_machine_score).sum() + 1e-7)
            
            clone_machine_summary = highlight_seg_machine_score.clone().detach().cpu()
            clone_machine_summary = clone_machine_summary.numpy()
            aP50 = average_precision_score(gt_top50_summary, clone_machine_summary)
            aP15 = average_precision_score(gt_top15_summary, clone_machine_summary)
            map50_history.append(aP50)
            map15_history.append(aP15)

        final_f_score = np.mean(fscore_history)
        final_map50 = np.mean(map50_history)
        final_map15 = np.mean(map15_history)

        return final_f_score, final_map50, final_map15

    def test(self, ckpt_path):
        if ckpt_path != None:
            print("Testing Model: ", ckpt_path)
            print("Device: ",  self.config.device)
            self.model.load_state_dict(torch.load(ckpt_path))
        
        test_fscore, test_map50, test_map15 = self.evaluate(dataloader=self.test_loader)

        print("------------------------------------------------------")
        print(f"   TEST RESULT on {ckpt_path}: ")
        print('   TEST MRHiSum F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5}'.format(test_fscore, test_map50, test_map15))
        print("------------------------------------------------------")
        
        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write("Testing on Model " + ckpt_path + '\n')
        f.write('Test F-score ' + str(test_fscore) + '\n')
        f.write('Test MAP50   ' + str(test_map50) + '\n')
        f.write('Test MAP15   ' + str(test_map15) + '\n\n')
        f.flush()
    
    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """ Initialize 'net' network weights, based on the chosen 'init_type' and 'init_gain'.

        :param nn.Module net: Network to be initialized.
        :param str init_type: Name of initialization method: normal | xavier | kaiming | orthogonal.
        :param float init_gain: Scaling factor for normal.
        """
        for name, param in net.named_parameters():
            if 'weight' in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))  # ReLU activation function
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))      # ReLU activation function
                else:
                    raise NotImplementedError(f"initialization method {init_type} is not implemented.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

if __name__ == '__main__':
    pass