import torch
from efficientnet_pytorch import EfficientNet
from torch import nn
from transformers import (
    AdamW,
    AutoModel,
    DebertaForMaskedLM,
    get_linear_schedule_with_warmup,
)

import pytorch_lightning as pl


class VQAModel(nn.Module):
    def __init__(
        self,
        cnn_model="efficientnet-b0",
        bert_model="microsoft/deberta-base",
        colorbert="",
        hidden_size=1000,
        num_labels=430,
        dropout=0.5,
    ):
        super().__init__()
        self.cnn = EfficientNet.from_pretrained(cnn_model, include_top=False)
        if colorbert:
            print("Loading colorbert")
            self.bert = DEBERTA.load_from_checkpoint(colorbert).bert.deberta
        else:
            self.bert = AutoModel.from_pretrained(bert_model)
        self.decoder = nn.Sequential(
            nn.Linear(1280 + 768, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, images, tokens):
        cnn_out = self.cnn(images).squeeze(-1).squeeze(-1)
        # cnn_out.shape == (batch_size, 1280)
        bert_out = self.bert(**tokens)
        cls_tokens = bert_out["last_hidden_state"][:, 0, :]
        # cls_tokens.shape == (batch_size, 768)
        combined = torch.cat((cnn_out, cls_tokens), dim=-1)
        # combined.shape == (batch_size, 1280 + 768)
        out = self.decoder(combined)
        # out.shape == (batch_size, 430)
        return out


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DEBERTA(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.lr = args.learning_rate
        self.epochs = args.epochs
        self.steps_per_epoch = args.evaluate_every
        self.bert = DebertaForMaskedLM.from_pretrained(args.model_name)

    def forward(self, masked, labels=None):
        return self.bert(input_ids=masked, labels=labels)

    """
    def default_step(self, batch, batch_idx, mode):
        assert mode in {"train", "val", "test"}
        masked, labels = batch
        outputs = self.forward(masked, labels)
        loss = outputs["loss"]
        logs = {f'{mode}_loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_epoch=True, prog_bar=True, logger=True)
        return loss
    """

    def training_step(self, batch, batch_idx):
        masked, labels = batch
        outputs = self.forward(masked, labels)
        loss = outputs["loss"]
        logs = {"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"]}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        masked, labels = batch
        outputs = self.forward(masked, labels)
        loss = outputs["loss"]
        logs = {"val_loss": loss}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        masked, labels = batch
        outputs = self.forward(masked, labels)
        loss = outputs["loss"]
        logs = {"test_loss": loss}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=self.lr)
        warmup_steps = self.steps_per_epoch // 2
        total_steps = self.steps_per_epoch * self.epochs - warmup_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
