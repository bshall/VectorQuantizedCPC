import argparse
from pathlib import Path
import json
import numpy as np
import torch
from model import Encoder
from tqdm import tqdm


def encode_dataset(args, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Encoder(in_channels=params["preprocessing"]["num_mels"],
                    encoder_channels=params["model"]["encoder_channels"],
                    z_dim=params["model"]["z_dim"],
                    c_dim=params["model"]["c_dim"])
    model.to(device)

    print("Load checkpoint from: {}:".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    hop_length_seconds = params["preprocessing"]["hop_length"] / params["preprocessing"]["sample_rate"]

    in_dir = Path(args.in_dir)
    for path in tqdm(in_dir.rglob("*.mel.npy")):
        mel = torch.from_numpy(np.load(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            z, c, _, _ = model(mel)

        output = z.squeeze().cpu().numpy()
        time = np.linspace(0, (mel.size(-1) - 1) * hop_length_seconds, len(output))
        relative_path = path.relative_to(in_dir).with_suffix("")
        out_path = out_dir / relative_path
        out_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(out_path.with_suffix(".npz"), features=output, time=time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path to resume")
    parser.add_argument("--in-dir", type=str, help="Directory to encode")
    parser.add_argument("--out-dir", type=str, help="Output path")
    args = parser.parse_args()
    with open("config.json") as file:
        params = json.load(file)
    encode_dataset(args, params)
