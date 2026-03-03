#!/usr/bin/env python3
"""Generate a DJ mix using v2 stem-aware transitions."""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidj.analyzer import analyze_track, analyze_library
from aidj.selector import build_playlist
from aidj.preprocessor import prepare_pair
from aidj.transition import TransitionGenerator
from aidj.assembler import assemble_mix, save_mix
from aidj import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str,
                        help="Directory of audio files or space-separated file paths",
                        nargs="+")
    parser.add_argument("--output", type=str, default="output/mix.wav",
                        help="Output file path")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained StemTransitionNet checkpoint")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache", type=str, default=None,
                        help="Analysis cache file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    # Collect input files
    input_paths = []
    for inp in args.input:
        if os.path.isdir(inp):
            for root, _, files in os.walk(inp):
                for f in sorted(files):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in config.AUDIO_EXTENSIONS:
                        input_paths.append(os.path.join(root, f))
        elif os.path.isfile(inp):
            input_paths.append(inp)

    if len(input_paths) < 2:
        print("Need at least 2 tracks to generate a mix.")
        sys.exit(1)

    print(f"Found {len(input_paths)} tracks")

    # Analyze tracks
    print("Analyzing tracks...")
    tracks = []
    for path in input_paths:
        try:
            info = analyze_track(path)
            tracks.append(info)
            print(f"  {os.path.basename(path)}: {info.bpm:.0f} BPM, {info.key}")
        except Exception as e:
            log.warning(f"Failed to analyze {path}: {e}")

    if len(tracks) < 2:
        print("Not enough analyzable tracks.")
        sys.exit(1)

    # Build playlist
    print("\nBuilding playlist...")
    playlist = build_playlist(tracks)
    for i, t in enumerate(playlist):
        print(f"  {i+1}. {os.path.basename(t.path)} "
              f"({t.bpm:.0f} BPM, {t.camelot_code or t.key})")

    # Initialize transition generator
    generator = TransitionGenerator(model_path=args.model, device=args.device)

    # Generate transitions
    print("\nGenerating transitions...")
    playlist_audio = []
    transitions = []
    cue_outs = []
    cue_ins = []

    for i, track in enumerate(playlist):
        audio, _ = librosa.load(track.path, sr=config.SR, mono=True)
        playlist_audio.append(audio)

    for i in range(len(playlist) - 1):
        track_a = playlist[i]
        track_b = playlist[i + 1]

        print(f"  Transition {i+1}: {os.path.basename(track_a.path)} -> "
              f"{os.path.basename(track_b.path)}")

        audio_a, audio_b, cue_out, cue_in = prepare_pair(track_a, track_b)

        transition, _ = generator.generate(audio_a, audio_b, cue_out, cue_in)
        transitions.append(transition)
        cue_outs.append(cue_out)
        cue_ins.append(cue_in)

    # Add dummy cue points for first/last tracks
    # cue_outs needs entries for all tracks, cue_ins needs entries for all tracks
    final_cue_outs = []
    final_cue_ins = []
    for i in range(len(playlist)):
        if i < len(cue_outs):
            final_cue_outs.append(cue_outs[i])
        else:
            final_cue_outs.append(playlist[i].duration)

        if i > 0 and i - 1 < len(cue_ins):
            final_cue_ins.append(cue_ins[i - 1])
        else:
            final_cue_ins.append(0.0)

    # Assemble
    print("\nAssembling mix...")
    mix = assemble_mix(playlist_audio, transitions, final_cue_outs, final_cue_ins)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_mix(mix, args.output)
    duration_min = len(mix) / config.SR / 60
    print(f"\nMix saved to {args.output} ({duration_min:.1f} minutes)")


if __name__ == "__main__":
    main()
