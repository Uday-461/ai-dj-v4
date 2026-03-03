import json
import logging
from pathlib import Path
from aidj import config

log = logging.getLogger(__name__)

def load_dataset(path=None):
    """Load the full djmix-dataset JSON."""
    path = path or config.DATASET_JSON
    with open(path) as f:
        return json.load(f)

def select_subset(dataset, size=None, min_transitions=3, min_track_coverage=0.5):
    """Select a subset of mixes suitable for training.

    Filters:
    - audio_source in ("soundcloud", "youtube")
    - num_available_transitions >= min_transitions
    - At least min_track_coverage fraction of tracks have non-null YouTube IDs

    Sorts by num_available_transitions descending, takes top `size` mixes.
    Returns list of mix dicts with an added 'genres' field extracted from tags.
    """
    size = size or config.SUBSET_SIZE
    filtered = []
    for mix in dataset:
        if mix.get("audio_source") not in ("soundcloud", "youtube"):
            continue
        if mix.get("num_available_transitions", 0) < min_transitions:
            continue
        tracks = mix.get("tracklist", [])
        if not tracks:
            continue
        available = sum(1 for t in tracks if t.get("id") is not None)
        if available / len(tracks) < min_track_coverage:
            continue
        mix_copy = dict(mix)
        mix_copy["genres"] = [
            t["key"].removeprefix("Category:") if isinstance(t, dict) else t
            for t in mix.get("tags", [])
        ]
        filtered.append(mix_copy)

    filtered.sort(key=lambda m: m["num_available_transitions"], reverse=True)
    subset = filtered[:size]

    log.info(f"Selected {len(subset)} mixes from {len(dataset)} total "
             f"({len(filtered)} passed filters)")
    return subset

def save_manifest(subset, output_path):
    """Save the selected subset as a manifest JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "num_mixes": len(subset),
        "num_tracks": sum(len(m["tracklist"]) for m in subset),
        "num_transitions": sum(m["num_available_transitions"] for m in subset),
        "mixes": subset,
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Manifest saved to {output_path}")
    return manifest

def print_stats(manifest):
    """Print summary statistics."""
    print(f"Mixes:       {manifest['num_mixes']}")
    print(f"Tracks:      {manifest['num_tracks']}")
    print(f"Transitions: {manifest['num_transitions']}")

    # Genre distribution
    genre_counts = {}
    for mix in manifest["mixes"]:
        for g in mix.get("genres", []):
            genre_counts[g] = genre_counts.get(g, 0) + 1
    top_genres = sorted(genre_counts.items(), key=lambda x: -x[1])[:10]
    print(f"\nTop genres:")
    for genre, count in top_genres:
        print(f"  {genre}: {count}")
