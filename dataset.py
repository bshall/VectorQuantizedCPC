import numpy as np
import torch
from torch.utils.data import Dataset
import json
import random
from pathlib import Path


class CPCDataset(Dataset):
    def __init__(self, root, n_sample_frames, n_utterances_per_speaker, hop_length, sr):
        self.root = Path(root)
        self.n_sample_frames = n_sample_frames
        self.n_utterances_per_speaker = n_utterances_per_speaker

        with open(self.root / "speakers.json") as file:
            self.speakers = sorted(json.load(file))

        min_duration = n_sample_frames * hop_length / sr
        with open(self.root / "train.json") as file:
            metadata = json.load(file)
        metadata_by_speaker = dict()
        for _, _, duration, out_path in metadata:
            if duration > min_duration:
                out_path = Path(out_path)
                speaker = out_path.parent.stem
                metadata_by_speaker.setdefault(speaker, []).append(out_path)
        self.metadata = [
            (k, v) for k, v in metadata_by_speaker.items()
            if len(v) >= n_utterances_per_speaker]

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


class WavDataset(Dataset):
    def __init__(self, root, hop_length, sr, sample_frames):
        self.root = Path(root)
        self.hop_length = hop_length
        self.sample_frames = sample_frames

        with open(self.root / "speakers.json") as file:
            self.speakers = sorted(json.load(file))

        min_duration = (sample_frames + 2) * hop_length / sr
        with open(self.root / "train.json") as file:
            metadata = json.load(file)
            self.metadata = [
                Path(out_path) for _, _, duration, out_path in metadata
                if duration > min_duration
            ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        path = self.root.parent / path

        audio = np.load(path.with_suffix(".wav.npy"))
        mel = np.load(path.with_suffix(".mel.npy"))

        pos = random.randint(0, mel.shape[-1] - self.sample_frames - 2)
        mel = mel[:, pos:pos + self.sample_frames]
        audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]

        speaker = self.speakers.index(path.parts[-2])

        return torch.LongTensor(audio), torch.FloatTensor(mel), speaker
