from torch.utils.data import DataLoader
from gv_tools.util.logger import Logger
import numpy as np
from torch.optim import lr_scheduler
import torch
from tqdm import tqdm
import os
from data.data_process_date import DataProcessor
from data.data_loader_extra import BasicData
from torch.autograd import Variable
from modules.models.termcast import T


class Trainer:
    def __init__(self, cfg_params, args, logger: Logger, result_logger: Logger):
        self.cfg = cfg_params
        cfg_params.copyAttrib(self)
        self.__dict__.update(args.__dict__)
        self.logger = logger
        self.res_logger = result_logger
        self.data_processor = DataProcessor(cfg=self.cfg)
        self.train_dataloader = None
        self.val_dataloader = None
        self.scaler = self.data_processor.scaler
        self.net = None
        self.min_val_loss = float('inf')

    def build_data_loader(self):
        self.logger.log('... Start Building Data Loaders ...')
        x_c_np_train, x_p_np_train, x_t_np_train, y_train, extra_train = self.data_processor.get_train_data()
        self.logger.field("Total training samples", len(x_c_np_train))
        train_dataset = BasicData(x_c_np_train, x_p_np_train, x_t_np_train, y_train, extra_train, self.scaler)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        x_c_np_val, x_p_np_val, x_t_np_val, y_val, extra_val = self.data_processor.get_val_data()
        self.logger.field("Total validation samples", len(x_c_np_val))
        val_dataset = BasicData(x_c_np_val, x_p_np_val, x_t_np_val, y_val, extra_val, self.scaler)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.logger.log("... Data Loaders Done ...")

    def build_model(self, pretrained):
        self.logger.log('... Building model network ...')
        self.net = T(self.cfg)

        if self.use_cuda:
            self.net.cuda()
            self.logger.log('USE_GPU = True')

        if pretrained is not None:
            self.logger.log("Loading Pretrained Model")
            self.net.load_state_dict(torch.load(pretrained))

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr)
        if self.lr_decay_value > 0.0:
            self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                               factor=self.lr_decay_value,
                                                               patience=self.lr_decay_patience,
                                                               verbose=True)
        self.logger.log("... Network Build ...")

    def val(self):
        self.net.eval()
        val_loss = []
        for i, (x_closeness, x_period, x_trend, data_y, extra) in enumerate(self.val_dataloader):

            if self.use_cuda:
                x_closeness, x_period, x_trend, extra = Variable(
                    x_closeness).cuda(), Variable(
                    x_period).cuda(), Variable(
                    x_trend).cuda(), Variable(extra).cuda()
            else:
                x_closeness, x_period, x_trend, extra = Variable(
                    x_closeness), Variable(
                    x_period), Variable(
                    x_trend), Variable(extra)
            out, predict_relation, infer_relation = self.net(x_closeness, x_period, x_trend, extra)
            loss = self.loss_function(out, data_y, predict_relation, infer_relation)
            val_loss.append(loss.item())
        self.logger.field('Val loss', np.mean(val_loss))

        return np.mean(val_loss)

    def save_model(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        torch.save(self.net.state_dict(),
                   os.path.join(self.save_path, 'params.pkl'))

    def train(self):
        self.logger.log("... Training Started ...")
        epoch = 0

        while epoch < self.epochs:
            epoch += 1
            losses = []
            self.net.train()
            for i, (x_closeness, x_period, x_trend, data_y, extra) in tqdm(enumerate(self.train_dataloader)):

                if self.use_cuda:
                    x_closeness, x_period, x_trend, data_y, extra = Variable(
                        x_closeness).cuda(), Variable(
                        x_period).cuda(), Variable(
                        x_trend).cuda(), Variable(data_y).cuda(), Variable(extra).cuda()
                else:
                    x_closeness, x_period, x_trend, data_y, extra = Variable(
                        x_closeness), Variable(
                        x_period), Variable(
                        x_trend), Variable(data_y), Variable(extra)

                self.optimizer.zero_grad()
                out, predict_relation, infer_relation = self.net(x_closeness, x_period, x_trend, extra)
                loss = self.loss_function(out, data_y, predict_relation, infer_relation)
                loss.backward()
                # update the weights
                self.optimizer.step()
                losses.append(loss.item())

            if epoch % self.verbose_step == 0:
                self.logger.field('Epoch', epoch)
                self.logger.field('loss', np.mean(losses))
            if epoch % self.interval_val_epochs == 0 or epoch == self.epochs:
                self.logger.field('Val Epoch', epoch)
                val_loss = self.val()
                if val_loss < self.min_val_loss:
                    self.min_val_loss = val_loss
                    self.save_model()
                if self.lr_decay_value > 0.0:
                    self.lr_scheduler.step(val_loss)

    @staticmethod
    def loss_function(pred_y, gt_y, predict_relation, infer_relation):
        loss_fn = torch.nn.CosineEmbeddingLoss()
        y = torch.ones(predict_relation.size()[0]).cuda()
        mse_loss = torch.nn.functional.mse_loss(pred_y, gt_y)
        relation_loss = loss_fn(predict_relation, infer_relation, y)

        return mse_loss + relation_loss


