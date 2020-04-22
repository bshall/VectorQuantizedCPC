from pathlib import Path
import json
import argparse
import sys
import numpy as np
from dataset import SpeakerDataset
import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.version()
import torch.optim as optim
from model import Encoder, CPCLoss
import os
from os import path
from itertools import chain
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import apex.amp as amp
# import apex.apex as apex
# sys.path.append(path.join(".", "apex", "apex", "amp"))
# import apex.apex.amp.amp as amp

from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(model, cpc, optimizer, amp, epoch, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cpc": cpc.state_dict(),
        "amp": amp.state_dict(),
        "epoch": epoch}
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(epoch)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))


def train_fn(args, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Encoder(in_channels=params["preprocessing"]["num_mels"],
                    encoder_channels=params["model"]["encoder_channels"],
                    z_dim=params["model"]["z_dim"],
                    c_dim=params["model"]["c_dim"])
    model.to(device)

    cpc = CPCLoss(n_speakers=params["training"]["n_speakers"],
                  n_prediction_steps=params["training"]["n_prediction_steps"],
                  n_utterances_per_speaker=params["training"]["n_utterances_per_speaker"],
                  n_negatives=params["training"]["n_negatives"],
                  z_dim=params["model"]["z_dim"],
                  c_dim=params["model"]["c_dim"])
    cpc.to(device)

    optimizer = optim.Adam(chain(model.parameters(), cpc.parameters()),
                           lr=params["training"]["learning_rate"])

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.resume is not None:
        print("Resume checkpoint from: {}:".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        cpc.load_state_dict(checkpoint["cpc"])
        amp.load_state_dict(checkpoint["amp"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 1

    dataset = SpeakerDataset(root=args.data_dir,
                             n_sample_frames=params["training"]["n_sample_frames"] +
                                             params["training"]["n_prediction_steps"],
                             n_utterances_per_speaker=params["training"]["n_utterances_per_speaker"],
                             hop_length=params["preprocessing"]["hop_length"],
                             sample_rate=params["preprocessing"]["sample_rate"])

    dataloader = DataLoader(dataset, batch_size=params["training"]["n_speakers"],
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)

    for epoch in range(start_epoch, params["training"]["n_epochs"] + 1):
        average_loss = average_vq_loss = average_perplexity = 0
        average_accuracies = np.zeros(params["training"]["n_prediction_steps"])

        for i, (mels, _) in enumerate(tqdm(dataloader), 1):

            mels = mels.to(device)
            mels = mels.view(
                params["training"]["n_speakers"]*params["training"]["n_utterances_per_speaker"],
                params["preprocessing"]["num_mels"], -1)

            z, c, vq_loss, perplexity = model(mels, False)

            loss, accuracy = cpc(z, c)
            loss = loss #+ vq_loss

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            optimizer.step()

            average_loss += (loss.item() - average_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i
            average_accuracies += (np.array(accuracy) - average_accuracies) / i

        print("******************************************************************************")
        print("epoch: {}".format(epoch))
        print("loss: {}".format(average_loss))
        print("vq: {}".format(average_vq_loss))
        print("perplexity: {}".format(average_perplexity))
        print(average_accuracies)

        if epoch % params["training"]["checkpoint_interval"] == 0:
            save_checkpoint(model, cpc, optimizer, amp, epoch, Path(args.checkpoint_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=6, help="Number of dataloader workers.")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to save checkpoints.")
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    args = parser.parse_args()
    with open("config.json") as file:
        params = json.load(file)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    train_fn(args, params)
