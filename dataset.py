import numpy as np
import torch
import csv
import random
from torch.utils.data import Dataset
from pathlib import Path


class SpeakerDataset(Dataset):
    def __init__(self, root, n_sample_frames, n_utterances_per_speaker, hop_length, sample_rate):
        self.root = Path(root)
        self.n_sample_frames = n_sample_frames
        self.n_utterances_per_speaker = n_utterances_per_speaker

        with open(self.root / "speakers.csv") as file:
            reader = csv.reader(file)
            self.speakers = sorted([speaker for speaker, in reader])

        min_duration = n_sample_frames * hop_length / sample_rate
        with open(self.root / "train.csv") as file:
            reader = csv.reader(file)
            metadata = dict()
            for _, _, duration, out_path in reader:
                if float(duration) > min_duration:
                    out_path = Path(out_path)
                    speaker = out_path.stem.split("_")[0]
                    metadata.setdefault(speaker, []).append(out_path)
            self.metadata = [(k, v) for k, v in metadata.items() if len(v) >= n_utterances_per_speaker]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        speaker, paths = self.metadata[index]

        mels = list()
        paths = random.sample(paths, self.n_utterances_per_speaker)
        for path in paths:
            path = self.root.parent / path
            mel = np.load(path.with_suffix(".mel.npy"))
            pos = random.randint(0, mel.shape[1] - self.n_sample_frames)
            mel = mel[:, pos:pos + self.n_sample_frames]
            mels.append(mel)
        mels = np.stack(mels)

        return torch.from_numpy(mels), self.speakers.index(speaker)
