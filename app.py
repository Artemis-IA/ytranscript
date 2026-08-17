import os
import re
import tempfile
import warnings

import gradio as gr
import spaces
import torch
import yt_dlp
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")

# Configuration
MODEL_GPU = "large-v3-turbo"
MODEL_CPU = "small"
SUPPORTED_LANGUAGES = ["auto", "en", "fr", "es", "de", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ar", "hi"]


def _download_audio(url: str, out_dir: str, cookies_file: str | None = None) -> str:
    """Télécharge l'audio d'une URL YouTube (ou autre supporté par yt-dlp)."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "audio.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
        # Twitch: éviter le téléchargement des chunks en live
        "noplaylist": True,
        # Limite pour les VODs Twitch très longs
        "playlistend": 1,
        # Contourner la détection bot YouTube
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            },
        },
    }
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        base = os.path.join(out_dir, "audio")
        # yt-dlp génère audio.wav grâce au postprocessor
        if os.path.exists(base + ".wav"):
            return base + ".wav"
        # Fallback : cherche le premier fichier audio dans out_dir
        for f in os.listdir(out_dir):
            if f.startswith("audio"):
                return os.path.join(out_dir, f)
        raise FileNotFoundError("Audio file not found after download")


def _format_srt(segments) -> str:
    """Formate les segments au format SRT."""
    def _srt_time(seconds: float) -> str:
        millis = int((seconds % 1) * 1000)
        secs = int(seconds) % 60
        mins = (int(seconds) // 60) % 60
        hours = int(seconds) // 3600
        return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_time(seg.start)} --> {_srt_time(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


@spaces.GPU
def _transcribe(
    source_url: str | None,
    source_file: str | None,
    language: str,
    model_choice: str,
    output_format: str,
    cookies_file: str | None = None,
    progress=gr.Progress(),
):
    """Pipeline principal : téléchargement + transcription."""
    if not source_url and not source_file:
        return "Erreur : fournis une URL YouTube, Twitch, ou un fichier audio/vidéo.", "", ""

    # Choix du modèle
    has_gpu = torch.cuda.is_available()
    device = "cuda" if has_gpu and model_choice == "auto" else ("cuda" if model_choice == "gpu" else "cpu")
    compute_type = "int8" if device == "cuda" else "int8"
    model_name = MODEL_GPU if device == "cuda" else MODEL_CPU

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Étape 1 : obtenir le fichier audio
            if source_file:
                audio_path = source_file
            else:
                progress(0.1, desc="Téléchargement audio...")
                audio_path = _download_audio(source_url, tmp_dir, cookies_file)

            # Étape 2 : charger le modèle
            progress(0.3, desc=f"Chargement du modèle {model_name} sur {device}...")
            model = WhisperModel(model_name, device=device, compute_type=compute_type)

            # Étape 3 : transcription
            progress(0.5, desc="Transcription en cours...")
            lang = None if language == "auto" else language
            segments, info = model.transcribe(
                audio_path,
                language=lang,
                task="transcribe",
                word_timestamps=False,
                condition_on_previous_text=True,
                vad_filter=True,
            )

            # Étape 4 : formatage
            progress(0.8, desc="Formatage...")
            text_lines = []
            seg_list = []
            for seg in segments:
                seg_list.append(seg)
                text_lines.append(seg.text.strip())

            srt_text = _format_srt(seg_list)
            plain_text = "\n".join(text_lines)

            meta = f"Langue détectée : {info.language} | Probabilité : {info.language_probability:.2f} | Modèle : {model_name} | Device : {device}"
            return plain_text, srt_text, meta

        except Exception as e:
            return f"Erreur : {str(e)}", "", ""


def _detect_source(url):
    """Détecte la plateforme et extrait l'ID pour affichage."""
    if not url:
        return ""
    # YouTube: watch?v=XXXXX ou youtu.be/XXXXX
    yt_match = re.search(r"(?:v=|youtu\.be/)([0-9A-Za-z_-]{11})", url)
    if yt_match:
        return f"YouTube — ID : {yt_match.group(1)}"
    # Twitch VOD: twitch.tv/videos/1234567890
    twitch_vod = re.search(r"twitch\.tv/videos/(\d+)", url)
    if twitch_vod:
        return f"Twitch VOD — ID : {twitch_vod.group(1)}"
    # Twitch clip: clips.twitch.tv/ABC123 ou twitch.tv/clip/ABC123
    twitch_clip = re.search(r"(?:clips\.twitch\.tv/|twitch\.tv/\w+/clip/)([A-Za-z0-9_-]+)", url)
    if twitch_clip:
        return f"Twitch Clip — ID : {twitch_clip.group(1)}"
    # Twitch channel (live): twitch.tv/channelname
    twitch_live = re.search(r"twitch\.tv/([A-Za-z0-9_]{4,25})$", url)
    if twitch_live:
        return f"Twitch Live — Chaîne : {twitch_live.group(1)} (VOD uniquement)"
    return "Source personnalisée (yt-dlp)"


# Gradio UI
with gr.Blocks(title="yTranscript — YouTube to text", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ yTranscript\nColle une URL **YouTube**, **Twitch** (VOD/clip), ou uploade un fichier audio/vidéo pour obtenir un transcript.")

    with gr.Row():
        with gr.Column(scale=2):
            url_input = gr.Textbox(
                label="URL YouTube, Twitch, ou autre site supporté par yt-dlp",
                placeholder="https://www.youtube.com/watch?v=... ou https://www.twitch.tv/videos/...",
                lines=1,
            )
            url_status = gr.Textbox(label="", interactive=False, value="")
            url_input.change(_detect_source, inputs=url_input, outputs=url_status)

            file_input = gr.File(
                label="Ou upload un fichier audio/vidéo",
                file_types=["audio", "video"],
            )
            cookies_input = gr.File(
                label="Cookies (optionnel — cookies.txt pour YouTube)",
                file_types=[".txt"],
            )

        with gr.Column(scale=1):
            language = gr.Dropdown(
                choices=SUPPORTED_LANGUAGES,
                value="auto",
                label="Langue",
            )
            model_choice = gr.Radio(
                choices=["auto", "gpu", "cpu"],
                value="auto",
                label="Device / modèle",
                info="auto = GPU si dispo, sinon CPU. gpu force large-v3-turbo, cpu force small.",
            )
            output_format = gr.Radio(
                choices=["text", "srt", "both"],
                value="both",
                label="Format de sortie",
            )
            run_btn = gr.Button("Transcrire", variant="primary")

    with gr.Row():
        text_output = gr.Textbox(label="Texte", lines=20, show_copy_button=True)
        srt_output = gr.Textbox(label="SRT", lines=20, show_copy_button=True)

    meta_output = gr.Textbox(label="Métadonnées", interactive=False)

    run_btn.click(
        _transcribe,
        inputs=[url_input, file_input, language, model_choice, output_format, cookies_input],
        outputs=[text_output, srt_output, meta_output],
    )

    gr.Markdown("---\n*Propulsé par [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + [yt-dlp](https://github.com/yt-dlp/yt-dlp). Supporte YouTube, Twitch (VOD/clip), et 1300+ sites. Les Spaces HF gratuits sont en CPU : soyez patient pour les longues vidéos.*")

if __name__ == "__main__":
    demo.launch()
