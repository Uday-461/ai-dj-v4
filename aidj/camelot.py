from __future__ import annotations

# Camelot wheel: maps musical keys to Camelot codes for DJ-friendly key matching.
# Compatible keys: same code, ±1 number, or A↔B at same number.

KEY_TO_CAMELOT = {
    "C major": "8B",   "A minor": "8A",
    "G major": "9B",   "E minor": "9A",
    "D major": "10B",  "B minor": "10A",
    "A major": "11B",  "F# minor": "11A",
    "E major": "12B",  "C# minor": "12A",
    "B major": "1B",   "G# minor": "1A",
    "F# major": "2B",  "D# minor": "2A",
    "Db major": "3B",  "Bb minor": "3A",
    "Ab major": "4B",  "F minor": "4A",
    "Eb major": "5B",  "C minor": "5A",
    "Bb major": "6B",  "G minor": "6A",
    "F major": "7B",   "D minor": "7A",
    # Enharmonic aliases
    "Gb major": "2B",  "Eb minor": "2A",
    "C# major": "3B",  "A# minor": "3A",
    "G# major": "4B",
}

# Pitch class index (0-11, C=0) to key name
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


def pitch_class_to_key(pitch_class: int, is_minor: bool) -> str:
    name = _PITCH_CLASSES[pitch_class % 12]
    mode = "minor" if is_minor else "major"
    return f"{name} {mode}"


def key_to_camelot(key: str) -> str | None:
    return KEY_TO_CAMELOT.get(key)


def _parse_camelot(code: str) -> tuple[int, str]:
    number = int(code[:-1])
    letter = code[-1]
    return number, letter


def is_compatible(code_a: str, code_b: str) -> bool:
    if code_a is None or code_b is None:
        return False
    num_a, let_a = _parse_camelot(code_a)
    num_b, let_b = _parse_camelot(code_b)
    # Same code
    if num_a == num_b and let_a == let_b:
        return True
    # Same number, different letter (A↔B)
    if num_a == num_b and let_a != let_b:
        return True
    # ±1 on number, same letter (wraps 12→1)
    if let_a == let_b:
        diff = abs(num_a - num_b)
        if diff == 1 or diff == 11:
            return True
    return False
