from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.optim import Adam
import wandb
import yaml
import argparse

from models import SBERT_with_KLUE_BERT
from datasets import KorSTSDatasets, KorSTS_collate_fn, bucket_pair_indices


def main(config):

    wandb_config = dict (
        architecture = "klue/bert-base",
        batch_size=config.batch_size,
        learning_rate=config.lr,
    )
    run = wandb.init(project="sentence_bert", entity="intrandom5", config=wandb_config, name=config.log_name)

    train_datasets = KorSTSDatasets(config.train_x_dir, config.train_y_dir)
    valid_datasets = KorSTSDatasets(config.valid_x_dir, config.valid_y_dir)

    train_seq_lengths = []
    for s1, s2 in train_datasets.x:
        train_seq_lengths.append((len(s1), len(s2)))
    train_sampler = bucket_pair_indices(train_seq_lengths, batch_size=16, max_pad_len=10)

    train_loader = DataLoader(
        train_datasets, 
        collate_fn=KorSTS_collate_fn, 
        batch_sampler=train_sampler
    )
    valid_loader = DataLoader(
        valid_datasets,
        batch_size=16
    )

    model = SBERT_with_KLUE_BERT()

    epochs = 1
    criterion = nn.MSELoss()
    optimizer = Adam(params=model.parameters(), lr=.2e-5)

    train_loss = []
    valid_loss = []

    for epoch in range(epochs):
        for iter, data in enumerate(train_loader):
            s1, s2, label = data
            logits = model(s1, s2)
            loss = criterion(logits, label)
            train_loss.append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": train_loss})

        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                s1, s2, label = data
                logits = model(s1, s2)
                loss = criterion(logits, label)
                val_loss += loss
        valid_loss.append(val_loss.detach()/i)
        wandb.log({"valid_loss": valid_loss, "epoch": epoch})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training SBERT with klue/bert-base.')
    parser.add_argument("--config_path", type=str, help="config file path(.yaml)")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config)
    