import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np


class Trainer:
    def __init__(self, network, training_params):
        """
        Initialize
        """
        pass

    def train(self, batch_sample, epoch, step):
        """ train
        :param
        """
        pass

    def training_mode(self):
        """ Training mode """
        self.net.train()

    def evaluate_mode(self):
        """ evaluate(test mode) """
        self.net.eval()

    def save_model(self, path):
        """ Save model """
        torch.save({
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.current_loss,
            }, path)

    def load_weights_v2(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.current_loss = checkpoint['loss']

    @staticmethod
    def count_parameters(model: [nn.Module]):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)