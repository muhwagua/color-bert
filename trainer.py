import os
from operator import itemgetter

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import answer2idx, make_loaders
from model import VQAModel, freeze_model


class Trainer:
    def __init__(self, config):
        self.epoch = 0
        self.config = config
        self.device = torch.device(config.device)
        self.train_loader, self.valid_loader, self.test_loader = make_loaders(
            config.data
        )
        bert_model = config.model.bert
        self.model = VQAModel(
            bert_model=bert_model,
            cnn_model=config.model.cnn,
            colorbert=config.model.colorbert,
        ).to(self.device)
        if config.model.freeze_cnn:
            freeze_model(self.model.cnn)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        log_path = os.path.join("logs", config.name)
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        self.writer = SummaryWriter(log_dir=log_path)
        save_path = os.path.join("checkpoints", config.name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.save_path = save_path

    def train(self):
        for _ in tqdm(range(self.config.num_epochs), leave=False):
            self.epoch += 1
            train_loss = self._train_epoch()
            valid_loss = self._valid_epoch()
            if self.epoch % self.config.log_interval == 0:
                self.log(train_loss, valid_loss)
            if self.epoch % self.config.save_interval == 0:
                self.save_checkpoint()
        self.save_checkpoint()

    def tokenize(self, questions):
        tokens = self.tokenizer(
            list(questions), padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        return tokens

    def _train_epoch(self):
        total_loss = 0
        self.model.train()
        for (questions, images, answers, _) in self.train_loader:
            assert torch.isnan(images).sum() == 0
            images = images.to(self.device)
            answers = answers.to(self.device)
            tokens = self.tokenize(questions)
            self.optimizer.zero_grad()
            preds = self.model(images, tokens)
            loss = F.cross_entropy(preds, answers)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _valid_epoch(self):
        total_loss = 0
        self.model.eval()
        for (questions, images, answers, types) in self.valid_loader:
            images = images.to(self.device)
            answers = answers.to(self.device)
            tokens = self.tokenize(questions)
            preds = self.model(images, tokens)
            loss = F.cross_entropy(preds, answers)
            total_loss += loss.item()
        return total_loss / len(self.valid_loader)

    def test(self):
        num_total = 0
        num_correct = 0
        for (questions, images, answers, types) in tqdm(self.test_loader):
            images = images.to(self.device)
            answers = answers.to(self.device)
            tokens = self.tokenize(questions)
            preds = self.model(images, tokens)
            # preds.shape == (batch_size, 430)
            pred_idx = preds.argmax(dim=1)
            num_correct += (pred_idx == answers).sum()
            num_total += len(answers)
        return num_correct / num_total

    def log(self, train_loss, valid_loss):
        self.writer.add_scalar("train/loss", train_loss, self.epoch)
        self.writer.add_scalar("valid/loss", valid_loss, self.epoch)
        tqdm.write(
            f"Epoch {self.epoch:3d} train_loss: {train_loss:7.4f}, valid_loss: {valid_loss:7.4f}"
        )

    def _log_step(self, loss, mode):
        stdout = f"{mode} "
        self.writer.add_scalar(f"{mode}/{key}", value, self.epoch)
        stdout += f"{key}: {value:7.4f}, "
        tqdm.write(stdout.strip()[:-1])

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f"{self.epoch}.pt")
        if not os.path.isfile(path):
            state_dict = {
                "epoch": self.epoch,
                "optimizer": self.optimizer.state_dict(),
                "model": self.model.state_dict(),
            }
            torch.save(state_dict, path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        for key, value in checkpoint.items():
            try:
                self.__dict__[key].load_state_dict(value)
            except AttributeError:
                self.__dict__[key] = value
            except KeyError:
                continue
