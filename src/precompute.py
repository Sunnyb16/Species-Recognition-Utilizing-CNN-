import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed

from src.audio_processing import get_signal_centered_chunks

from src.templates import (
    compute_overlap,
    filter_secondary_labels_for_chunk
)
from src.config import *

from functools import partial

def process_single_row(
    row,
    output_dir,
    templates,
    encode_labels,
    sr,
    duration,
    threshold_db,
    band_peak_rel_db,
    min_event_duration,
    merge_gap,
    only_strongest,
    fallback_to_regular_split,
):

    import os
    import numpy as np
    import soundfile as sf

    path = row["full_path"]

    if not isinstance(path, str) or not os.path.exists(path):
        return []

    try:
        result = get_signal_centered_chunks(
            audio_path=path,
            sr=sr,
            duration=duration,
            mono=True,
            only_strongest=only_strongest,
            fallback_to_regular_split=fallback_to_regular_split,
            threshold_db=threshold_db,
            band_peak_rel_db=band_peak_rel_db,
            min_event_duration=min_event_duration,
            merge_gap=merge_gap,
        )
    except Exception:
        return []

    chunks = result["chunks"]
    intervals = result["intervals"]
    used_fallback = result["used_fallback"]

    if len(chunks) == 0:
        return []

    primary_label = row["primary_label"]
    secondary_labels = row.get("secondary_labels", [])
    if not isinstance(secondary_labels, list):
        secondary_labels = []

    samples = []

    for i, chunk in enumerate(chunks):

        # time window
        if used_fallback:
            chunk_start = i * duration
            chunk_end = (i + 1) * duration
        else:
            start, end = intervals[i]
            center = (start + end) / 2
            chunk_start = center - duration / 2
            chunk_end = center + duration / 2

        overlap = compute_overlap(chunk_start, chunk_end, intervals)

        filtered_secondary, scores = filter_secondary_labels_for_chunk(
            chunk=chunk,
            secondary_labels=secondary_labels,
            templates=templates,
            candidate_overlap_sec=overlap,
            min_overlap=0.2,
            similarity_threshold=0.75,
            sr=sr,
        )

        final_labels = list(set([primary_label] + filtered_secondary))
        target = encode_labels(final_labels)

        filename = f"{os.path.splitext(os.path.basename(path))[0]}_chunk{i}_{np.random.randint(1e9)}.wav"
        chunk_path = os.path.join(output_dir, filename)

        sf.write(chunk_path, chunk, sr)

        samples.append({
            "chunk_path": chunk_path,
            "target": target,
            "primary_label": primary_label,
            "secondary_labels": filtered_secondary,
            "similarity_scores": scores,
            "source_path": path,
            "chunk_index": i,
            "overlap": overlap,
            "used_fallback": used_fallback,
            "band_low_hz": result["band_low_hz"],
            "band_high_hz": result["band_high_hz"],
        })

    return samples



def precompute_chunk_cache(
    df,
    output_dir,
    templates,
    encode_labels,
    sr,
    duration,
    threshold_db,
    band_peak_rel_db,
    min_event_duration,
    merge_gap,
    only_strongest=False,
    fallback_to_regular_split=True,
    n_jobs=0,   # control cores here
):

    os.makedirs(output_dir, exist_ok=True)

    process_fn = partial(
        process_single_row,
        output_dir=output_dir,
        templates=templates,
        encode_labels=encode_labels,
        sr=sr,
        duration=duration,
        threshold_db=threshold_db,
        band_peak_rel_db=band_peak_rel_db,
        min_event_duration=min_event_duration,
        merge_gap=merge_gap,
        only_strongest=only_strongest,
        fallback_to_regular_split=fallback_to_regular_split,
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_fn)(row)
        for _, row in tqdm(df.iterrows(), total=len(df))
    )

    # flatten list
    samples = [s for sublist in results for s in sublist]

    return samples

def get_rows_for_files(master_df, file_df):
    file_ids = set(file_df["file_id"])
    return master_df[master_df["file_id"].isin(file_ids)].copy()




def precompute_fold(
    fold,
    file_level_df,
    master_df,
    templates,
    encode_labels,
    base_output_dir="data_processed",
    duration=5,
):

    print(f"\n=== Processing Fold {fold} ===")

    fold_dir = os.path.join(base_output_dir, f"fold_{fold}")
    train_dir = os.path.join(fold_dir, "train")
    val_dir   = os.path.join(fold_dir, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_samples_path = os.path.join(fold_dir, "train_samples.pkl")
    val_samples_path   = os.path.join(fold_dir, "val_samples.pkl")

    # -----------------------------
    # Skip if already computed
    # -----------------------------
    if os.path.exists(train_samples_path) and os.path.exists(val_samples_path):
        print(f"Skipping fold {fold} (already exists)")

        with open(train_samples_path, "rb") as f:
            train_samples = pickle.load(f)

        with open(val_samples_path, "rb") as f:
            val_samples = pickle.load(f)

        print(f"Loaded cached: train={len(train_samples)}, val={len(val_samples)}")
        return train_samples, val_samples

    # -----------------------------
    # Split
    # -----------------------------
    train_files = file_level_df[
        (file_level_df["split_role"] == "trainval") &
        (file_level_df["cv_fold"] != fold)
    ]

    val_files = file_level_df[
        (file_level_df["split_role"] == "trainval") &
        (file_level_df["cv_fold"] == fold)
    ]

    train_rows = get_rows_for_files(master_df, train_files)
    val_rows   = get_rows_for_files(master_df, val_files)

    print(f"Train rows: {len(train_rows)} | Val rows: {len(val_rows)}")

    # -----------------------------
    # Precompute (USES CONFIG)
    # -----------------------------
    train_samples = precompute_chunk_cache(
        df=train_rows,
        output_dir=train_dir,
        templates=templates,
        encode_labels=encode_labels,
        sr=SR,
        duration=duration,
        **CHUNK_CONFIG
    )

    val_samples = precompute_chunk_cache(
        df=val_rows,
        output_dir=val_dir,
        templates=templates,
        encode_labels=encode_labels,
        sr=SR,
        duration=duration,
        **CHUNK_CONFIG
    )

    print(f"Generated: train={len(train_samples)}, val={len(val_samples)}")

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise ValueError(f"Fold {fold} produced empty dataset")

    with open(train_samples_path, "wb") as f:
        pickle.dump(train_samples, f)

    with open(val_samples_path, "wb") as f:
        pickle.dump(val_samples, f)

    print(f"Saved → {fold_dir}")

    return train_samples, val_samples