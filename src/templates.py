import numpy as np
import librosa
from collections import defaultdict


SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 300
FMAX = 8000


def compute_mean_spectrum(
    y,
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    fmin=FMIN,
    fmax=FMAX,
):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_power = np.abs(S) ** 2

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    valid = (freqs >= fmin) & (freqs <= fmax)

    spectrum = S_power[valid, :].mean(axis=1)

    spectrum = spectrum / (spectrum.sum() + 1e-12)

    return spectrum, freqs[valid]


def build_species_templates(
    df,
    audio_col="full_path",
    label_col="primary_label",
    min_examples=5,
    max_per_species=None,
    sr=SR,
):
    species_spectra = defaultdict(list)
    shared_freqs = None

    for species, group in df.groupby(label_col):
        if len(group) < min_examples:
            continue

        if max_per_species is not None:
            group = group.sample(
                n=min(len(group), max_per_species),
                random_state=42
            )

        for _, row in group.iterrows():
            try:
                y, _ = librosa.load(row[audio_col], sr=sr, mono=True)

                spectrum, freqs = compute_mean_spectrum(y, sr=sr)

                if shared_freqs is None:
                    shared_freqs = freqs

                species_spectra[species].append(spectrum)

            except Exception:
                continue

    templates = {}
    for species, specs in species_spectra.items():
        if len(specs) == 0:
            continue

        template = np.mean(np.stack(specs), axis=0)
        template = template / (template.sum() + 1e-12)

        templates[species] = template

    return templates, shared_freqs

def spectrum_similarity(a, b):
    # cosine similarity
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


#Comparing similarity to a chunk with feature map
def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def chunk_similarity_to_species(
    chunk,
    species,
    templates,
    sr=SR,
):
    if species not in templates:
        return None

    chunk_spec, _ = compute_mean_spectrum(chunk, sr=sr)
    sim = cosine_similarity(chunk_spec, templates[species])
    return sim

def filter_secondary_labels_for_chunk(
    chunk,
    secondary_labels,
    templates,
    candidate_overlap_sec,
    min_overlap=0.2,
    similarity_threshold=0.75,
    sr=SR,
):
    # Early exit: no labels or no meaningful signal
    if not secondary_labels or candidate_overlap_sec < min_overlap:
        return [], {}

    kept = []
    scores = {}

    for species in secondary_labels:
        template = templates.get(species)
        if template is None:
            continue

        sim = chunk_similarity_to_species(chunk, species, templates, sr=sr)
        scores[species] = sim

        if sim is not None and sim >= similarity_threshold:
            kept.append(species)

    return kept, scores

def compute_overlap(chunk_start, chunk_end, intervals):
    """
    Compute total overlap (in seconds) between a chunk and a list of signal intervals.
    """
    total_overlap = 0.0

    for start, end in intervals:
        overlap_start = max(chunk_start, start)
        overlap_end = min(chunk_end, end)

        if overlap_end > overlap_start:
            total_overlap += (overlap_end - overlap_start)

    return total_overlap