import os
import logging
import time
from pathlib import Path
from aidj import config

log = logging.getLogger(__name__)


def download_audio(url, path, format="bestaudio", cookies_from_browser=None,
                   limit_rate=None, sleep_interval=0, max_sleep_interval=0,
                   throttled_rate=None):
    """Download audio from a URL using yt-dlp.

    Skips if file already exists (resumable).

    Args:
        url: URL to download from.
        path: Output file path.
        format: yt-dlp format string.
        cookies_from_browser: Browser name to extract cookies from (e.g. 'chrome').
        limit_rate: Max download speed (e.g. '5M', '500k').
        sleep_interval: Min seconds to sleep between downloads.
        max_sleep_interval: Max seconds to sleep between downloads (random range).
        throttled_rate: Re-extract if speed drops below this (e.g. '100K').
    """
    if os.path.isfile(path):
        log.debug(f"Already exists: {path}")
        return True

    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise RuntimeError("yt-dlp is required. Install with: pip install yt-dlp")

    # Strip .mp3 from outtmpl so FFmpeg postprocessor doesn't double it
    outtmpl = path
    if outtmpl.endswith(".mp3"):
        outtmpl = outtmpl[:-4]

    params = {
        "format": format,
        "outtmpl": outtmpl,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "postprocessor_args": {
            "FFmpegExtractAudio": ["-ac", "1"],  # mono
        },
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "js_runtimes": {"node": {}},
        "remote_components": {"ejs:github"},
    }

    if cookies_from_browser:
        params["cookiesfrombrowser"] = (cookies_from_browser,)
    if limit_rate:
        params["ratelimit"] = _parse_rate(limit_rate)
    if sleep_interval:
        params["sleep_interval"] = sleep_interval
    if max_sleep_interval:
        params["max_sleep_interval"] = max_sleep_interval
    if throttled_rate:
        params["throttledratelimit"] = _parse_rate(throttled_rate)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with YoutubeDL(params) as ydl:
            ydl.download([url])
        log.info(f"Downloaded: {path}")
        return True
    except Exception as e:
        log.warning(f"Failed to download {url}: {e}")
        return False


def _parse_rate(rate_str):
    """Parse rate string like '5M', '500k' to bytes/sec."""
    if isinstance(rate_str, (int, float)):
        return int(rate_str)
    rate_str = rate_str.strip()
    multipliers = {'k': 1024, 'K': 1024, 'm': 1024**2, 'M': 1024**2,
                   'g': 1024**3, 'G': 1024**3}
    if rate_str[-1] in multipliers:
        return int(float(rate_str[:-1]) * multipliers[rate_str[-1]])
    return int(rate_str)


def download_mix(mix, data_root=None, **kwargs):
    """Download a mix audio file."""
    data_root = Path(data_root or config.DATA_ROOT)
    mix_path = str(data_root / "mixes" / f"{mix['id']}.mp3")
    url = mix.get("audio_url", "")
    if not url:
        log.warning(f"No audio URL for mix {mix['id']}")
        return False
    return download_audio(url, mix_path, **kwargs)


def download_track(track_id, data_root=None, **kwargs):
    """Download a track from YouTube by its ID."""
    if track_id is None:
        return False
    data_root = Path(data_root or config.DATA_ROOT)
    track_path = str(data_root / "tracks" / f"{track_id}.mp3")
    url = f"https://www.youtube.com/watch?v={track_id}"
    return download_audio(url, track_path, **kwargs)


def download_mix_and_tracks(mix, data_root=None, track_sleep=0, **kwargs):
    """Download a mix and all its tracks.

    Args:
        mix: Mix dict with 'id', 'audio_url', 'tracklist'.
        data_root: Root data directory.
        track_sleep: Seconds to sleep between track downloads.
        **kwargs: Passed to download_audio (cookies_from_browser, limit_rate, etc).

    Returns (mix_success, track_results) where track_results is a dict
    mapping track_id -> success bool.
    """
    mix_ok = download_mix(mix, data_root, **kwargs)
    track_results = {}
    for i, track in enumerate(mix.get("tracklist", [])):
        tid = track.get("id")
        if tid is not None:
            if i > 0 and track_sleep > 0:
                time.sleep(track_sleep)
            track_results[tid] = download_track(tid, data_root, **kwargs)
    return mix_ok, track_results
