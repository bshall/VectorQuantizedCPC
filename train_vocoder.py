from pathlib import Path
import json
import argparse
import sys
import numpy as np
from dataset import SpeakerDataset, SpeechDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from model import Encoder, CPCLoss, Vocoder
from itertools import chain
import torch
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(model, optimizer, step, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step}
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))



def train_fn(args, params):

    writer = SummaryWriter(Path("./runs") / args.checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Encoder(in_channels=params["preprocessing"]["num_mels"],
                    encoder_channels=params["model"]["encoder_channels"],
                    z_dim=params["model"]["z_dim"],
                    c_dim=params["model"]["c_dim"])
    model.to(device)

    print("Load checkpoint from: {}:".format(args.cpc_checkpoint))
    checkpoint = torch.load(args.cpc_checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # cpc = CPCLoss(n_speakers=params["training"]["n_speakers"],
    #               n_prediction_steps=params["training"]["n_prediction_steps"],
    #               n_utterances_per_speaker=params["training"]["n_utterances_per_speaker"],
    #               n_negatives=params["training"]["n_negatives"],
    #               z_dim=params["model"]["z_dim"],
    #               c_dim=params["model"]["c_dim"])
    # cpc.to(device)

    vocoder = Vocoder(in_channels=params["model"]["z_dim"],
                  num_speakers=params["model"]["vocoder"]["num_speakers"],
                  speaker_embedding_dim=params["model"]["vocoder"]["speaker_embedding_dim"],
                  conditioning_channels=params["model"]["vocoder"]["conditioning_channels"],
                  embedding_dim=params["model"]["vocoder"]["embedding_dim"],
                  rnn_channels=params["model"]["vocoder"]["rnn_channels"],
                  fc_channels=params["model"]["vocoder"]["fc_channels"],
                  bits=params["preprocessing"]["bits"],
                  hop_length=params["preprocessing"]["hop_length"])
    vocoder.to(device)

    optimizer = optim.Adam(vocoder.parameters(), lr=params["training"]["learning_rate"])



    if args.resume is not None:
        print("Resume checkpoint from: {}:".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        vocoder.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # amp.load_state_dict(checkpoint["amp"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    dataset = SpeechDataset(root=args.data_dir,
                            sample_frames=params["training"]["sample_frames"],
                            hop_length=params["preprocessing"]["hop_length"],
                            sample_rate=params["preprocessing"]["sample_rate"])

    dataloader = DataLoader(dataset, batch_size=params["training"]["batch_size"],
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)

    num_epochs = params["training"]["num_steps"] // len(dataloader) + 1
    start_epoch = global_step // len(dataloader) + 1

    for epoch in range(start_epoch, num_epochs + 1):
        average_loss = 0

        for i, (audio, mels, speakers) in enumerate(tqdm(dataloader), 1):

            audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)
            with torch.no_grad():
                z, c, _, _ = model(mels, False)

            output = vocoder(audio[:, :-1], c, speakers)
            recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])

            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            average_loss += (recon_loss.item() - average_loss) / i

            global_step += 1

            if global_step % params["training"]["global_checkpoint_interval"] == 0:
                save_checkpoint(vocoder, optimizer, global_step, Path(args.checkpoint_dir))

        writer.add_scalar("recon_loss/train", average_loss, global_step)
            
        print("******************************************************************************")
        print("epoch: {}".format(epoch))
        print("loss: {}".format(average_loss))

        # if epoch % params["training"]["checkpoint_interval"] == 0:
        #     save_checkpoint(model, cpc, optimizer, epoch, Path(args.checkpoint_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=6, help="Number of dataloader workers.")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to save checkpoints.")
    parser.add_argument("--cpc-checkpoint", type=str, default=None, help="Checkpoint path to restore CPC model")
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    args = parser.parse_args()
    with open("config.json") as file:
        params = json.load(file)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    train_fn(args, params)