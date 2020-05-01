import hydra
from hydra import utils
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import CPCDataset
from scheduler import WarmupScheduler
from model import Encoder, CPCLoss


def save_checkpoint(encoder, cpc, optimizer, scheduler, epoch, checkpoint_dir):
    checkpoint_state = {
        "encoder": encoder.state_dict(),
        "cpc": cpc.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(epoch)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))


@hydra.main(config_path="config/train_cpc.yaml")
def train_model(cfg):
    tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.checkpoint_dir
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    writer = SummaryWriter(tensorboard_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    cpc = CPCLoss(**cfg.model.cpc)
    encoder.to(device)
    cpc.to(device)

    optimizer = optim.Adam(
        chain(encoder.parameters(), cpc.parameters()),
        lr=cfg.training.scheduler.initial_lr)
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=cfg.training.scheduler.warmup_epochs,
        initial_lr=cfg.training.scheduler.initial_lr,
        max_lr=cfg.training.scheduler.max_lr,
        milestones=cfg.training.scheduler.milestones,
        gamma=cfg.training.scheduler.gamma)

    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        cpc.load_state_dict(checkpoint["cpc"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 1

    root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path
    dataset = CPCDataset(
        root=root_path,
        n_sample_frames=cfg.training.sample_frames + cfg.training.n_prediction_steps,
        n_utterances_per_speaker=cfg.training.n_utterances_per_speaker,
        hop_length=cfg.preprocessing.hop_length,
        sr=cfg.preprocessing.sr)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.n_speakers_per_batch,
        shuffle=True,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=True)

    for epoch in range(start_epoch, cfg.training.n_epochs + 1):
        if epoch % cfg.training.log_interval == 0 or epoch == start_epoch:
            average_cpc_loss = average_vq_loss = average_perplexity = 0
            average_accuracies = np.zeros(cfg.training.n_prediction_steps // 2)

        for i, (mels, _) in enumerate(tqdm(dataloader), 1):
            mels = mels.to(device)
            mels = mels.view(
                cfg.training.n_speakers_per_batch *
                cfg.training.n_utterances_per_speaker,
                cfg.preprocessing.n_mels, -1)

            optimizer.zero_grad()

            z, c, vq_loss, perplexity = encoder(mels)
            cpc_loss, accuracy = cpc(z, c)
            loss = cpc_loss + vq_loss

            loss.backward()
            optimizer.step()

            average_cpc_loss += (cpc_loss.item() - average_cpc_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i
            average_accuracies += (np.array(accuracy) - average_accuracies) / i

        scheduler.step()

        if epoch % cfg.training.log_interval == 0 and epoch != start_epoch:
            writer.add_scalar("cpc_loss/train", average_cpc_loss, epoch)
            writer.add_scalar("vq_loss/train", average_vq_loss, epoch)
            writer.add_scalar("perplexity/train", average_perplexity, epoch)

            print("epoch:{}, cpc loss:{:.2E}, vq loss:{:.2E}, perpexlity:{:.3f}"
                  .format(epoch, cpc_loss, average_vq_loss, average_perplexity))
            print(100 * average_accuracies)

        if epoch % cfg.training.checkpoint_interval == 0 and epoch != start_epoch:
            save_checkpoint(
                encoder, cpc, optimizer,
                scheduler, epoch, checkpoint_dir)


if __name__ == "__main__":
    train_model()
