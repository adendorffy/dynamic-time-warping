from utils.features import WordUnit
import random
from torchaudio.functional import resample
import torch
import torchaudio
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa


INT16_MAX = (2**15) - 1
hop_length = 320
sample_rate = 16000


def sample_files(dataset, sample_size=100):
    in_paths = list(dataset.in_dir.rglob(f"**/*{dataset.audio_ext}"))

    if sample_size == -1:
        return in_paths

    if sample_size > len(in_paths):
        sample_files = in_paths
    else:
        sample_files = random.sample(in_paths, sample_size)
    return sample_files


def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def get_encodings(dataset, model_name, sampled_paths, layer=7, save=False):
    words = []
    model = None

    model_pipelines = {
        "hubert_base": torchaudio.pipelines.HUBERT_BASE,
        "hubert_large": torchaudio.pipelines.HUBERT_LARGE,
        "hubert_xlarge": torchaudio.pipelines.HUBERT_XLARGE,
        "wavlm_base": torchaudio.pipelines.WAVLM_BASE,
        "wavlm_large": torchaudio.pipelines.WAVLM_LARGE,
        "wavlm_base_plus": torchaudio.pipelines.WAVLM_BASE_PLUS,
    }

    if model_name != "mfcc":
        bundle = model_pipelines.get(model_name, torchaudio.pipelines.HUBERT_BASE)
        model = bundle.get_model()
        model.eval()

    align_df = pd.read_csv(dataset.align_dir / "alignments.csv")

    word_id = 0
    for wav_path in tqdm(sampled_paths, desc="Getting units"):
        wav_df = align_df[align_df["filename"] == wav_path.stem]

        wav, sr = torchaudio.load(wav_path)
        wav = resample(wav, sr, 16000)

        if model_name == "mfcc":
            wav, sr = librosa.core.load(wav_path, sr=None)
            wav = preemphasis(wav, coeff=0.97)

        word, word_id = get_encoding(
            dataset,
            wav,
            model_name,
            model,
            layer,
            sr,
            wav_path,
            wav_df,
            word_id,
            save,
        )

        words.extend(word)

    return words


def get_encoding(
    dataset,
    wav,
    model_name,
    model,
    layer,
    sr,
    wav_path,
    wav_df,
    word_id,
    save=False,
):
    words = []
    if model and model_name != "mfcc":
        with torch.inference_mode():
            encoding, _ = model.extract_features(wav, num_layers=layer)

        encoding = encoding[layer - 1].squeeze().cpu().numpy()
    else:
        mfcc = librosa.feature.mfcc(
            y=wav,
            sr=sr,
            n_mfcc=13,
            n_mels=24,
            n_fft=int(np.floor(0.025 * sr)),
            hop_length=int(np.floor(0.01 * sr)),
            fmin=64,
            fmax=8000,
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
        encoding = np.hstack([mfcc.T, mfcc_delta.T, mfcc_delta_delta.T])

    for w in range(max(wav_df["word_id"])):
        word_df = wav_df[wav_df["word_id"] == w]

        if not isinstance(word_df["text"].iloc[0], str):
            true_word = "_"
        else:
            true_word = word_df["text"].iloc[0]

        new_word = WordUnit(
            id=word_id,
            filename=wav_path.stem,
            index=w,
            true_word=true_word,
            boundaries=[word_df["word_start"].iloc[0], word_df["word_end"].iloc[0]],
            discrete=True,
        )
        new_word.add_encoding_by_flags(encoding, None, False)

        word_id += 1
        words.append(new_word)

        if save:
            out_path = (
                dataset.feat_dir
                / model_name
                / wav_path.relative_to(dataset.in_dir).with_suffix("")
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)

            out_path = str(out_path) + f"_{w}.npy"
            np.save(out_path, new_word.clean_encoding)

    return words, word_id
