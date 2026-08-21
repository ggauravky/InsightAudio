import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    
    cookies_file = os.getenv("YOUTUBE_COOKIES_FILE")
    if not cookies_file and os.path.exists("cookies.txt"):
        cookies_file = "cookies.txt"

    base_opts = {
        "format": "ba/18/b",
        "outtmpl": output_path,
        "nocheckcertificate": True,
        "geo_bypass": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    if cookies_file and os.path.exists(cookies_file):
        base_opts["cookiefile"] = cookies_file

    client_strategies = [
        ["web_safari", "android", "web"],
        ["android", "mweb"],
        ["web", "android"],
    ]

    last_error = None
    for clients in client_strategies:
        try:
            ydl_opts = dict(base_opts)
            ydl_opts["extractor_args"] = {
                "youtube": {
                    "player_client": clients,
                }
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = (
                    ydl.prepare_filename(info)
                    .replace(".webm", ".wav")
                    .replace(".m4a", ".wav")
                    .replace(".mp4", ".wav")
                )
                return filename
        except Exception as e:
            last_error = e
            continue

    raise last_error or RuntimeError("Failed to download YouTube audio.")


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16khz
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...........")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
