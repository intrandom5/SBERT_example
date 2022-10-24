from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.optim import Adam
import wandb
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import os

from models import SBERT_with_KLUE_BERT
from datasets import KorSTSDatasets, KorSTS_collate_fn, bucket_pair_indices


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print("training on ", device)
    # wandb_config = dict (
    #     architecture = "klue/bert-base",
    #     batch_size=config['batch_size'],
    #     learning_rate=config['lr'],
    # )
    run = wandb.init(project="sentence_bert", entity="intrandom5", config=config, name=config['log_name'])

    train_datasets = KorSTSDatasets(config['train_x_dir'], config['train_y_dir'])
    valid_datasets = KorSTSDatasets(config['valid_x_dir'], config['valid_y_dir'])

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
        collate_fn=KorSTS_collate_fn,
        batch_size=16
    )

    model = SBERT_with_KLUE_BERT()
    if os.path.exists(config["model_load_path"]):
        model.load_state_dict(torch.load(config["model_load_path"]))
        print("weights loaded from", config["model_load_path"])
    else:
        print("no pretrained weights provided.")
    model.to(device)

    epochs = config['epochs']
    criterion = nn.MSELoss()
    optimizer = Adam(params=model.parameters(), lr=config['lr'])

    pbar = tqdm(range(epochs))

    for epoch in pbar:
        for iter, data in enumerate(tqdm(train_loader)):
            s1, s2, label = data
            s1 = s1.to(device)
            s2 = s2.to(device)
            label = label.to(device)
            
            logits = model(s1, s2)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach().item()
            wandb.log({"train_loss": loss})
            pbar.set_postfix({"train_loss": loss})

        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(valid_loader)):
                s1, s2, label = data
                s1 = s2.to(device)
                s2 = s2.to(device)
                label = label.to(device)
                
                logits = model(s1, s2)
                loss = criterion(logits, label)
                val_loss += loss.detach().item()
        val_loss = val_loss/i
        wandb.log({"valid_loss": val_loss, "epoch": epoch})
        pbar.set_postfix({"valid_loss": val_loss, "epoch": epoch})
        
    torch.save(model.state_dict(), config["model_save_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training SBERT with klue/bert-base.')
    parser.add_argument("--conf", type=str, help="config file path(.yaml)")
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config)
    