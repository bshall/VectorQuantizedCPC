import hydra
from hydra import utils
from pathlib import Path
from tqdm import tqdm

import apex.amp as amp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import WavDataset
from model import Encoder, Vocoder


def save_checkpoint(decoder, optimizer, amp, scheduler, step, checkpoint_dir):
    checkpoint_state = {
        "vocoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "amp": amp.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step}
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))


@hydra.main(config_path="config/train_vocoder.yaml")
def train_model(cfg):
    tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.checkpoint_dir
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    writer = SummaryWriter(tensorboard_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    vocoder = Vocoder(**cfg.model.vocoder)
    encoder.to(device)
    vocoder.to(device)

    optimizer = optim.Adam(
        vocoder.parameters(),
        lr=cfg.training.optimizer.lr)
    vocoder, optimizer = amp.initialize(vocoder, optimizer, opt_level="O1")
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.training.scheduler.milestones,
        gamma=cfg.training.scheduler.gamma)

    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        vocoder.load_state_dict(checkpoint["vocoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        amp.load_state_dict(checkpoint["amp"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    print("Resume cpc encoder from: {}:".format(cfg.cpc_checkpoint))
    encoder_path = utils.to_absolute_path(cfg.cpc_checkpoint)
    checkpoint = torch.load(encoder_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()

    root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path
    dataset = WavDataset(
        root=root_path,
        hop_length=cfg.preprocessing.hop_length,
        sr=cfg.preprocessing.sr,
        sample_frames=cfg.training.sample_frames)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=True)

    n_epochs = cfg.training.n_steps // len(dataloader) + 1
    start_epoch = global_step // len(dataloader) + 1

    for epoch in range(start_epoch, n_epochs + 1):
        average_loss = 0

        for i, (audio, mels, speakers) in enumerate(tqdm(dataloader), 1):
            audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                _, _, indices = encoder.encode(mels)
            output = vocoder(audio[:, :-1], indices, speakers)
            loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            optimizer.step()
            scheduler.step()

            average_loss += (loss.item() - average_loss) / i

            global_step += 1

            if global_step % cfg.training.checkpoint_interval == 0:
                save_checkpoint(
                    vocoder, optimizer, amp, scheduler, global_step, checkpoint_dir)

        writer.add_scalar("loss/train", average_loss, global_step)

        print("epoch:{}, loss:{:.3E}".format(epoch, average_loss))


if __name__ == "__main__":
    train_model()
