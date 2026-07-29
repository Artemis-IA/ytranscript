---
title: yTranscript
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
license: mit
---

# yTranscript

YouTube / Twitch to text — rapide, minimal, open-source.

Pile : `yt-dlp` + `faster-whisper` (`large-v3-turbo`) via Gradio.

## Sources supportées

- **YouTube** : vidéos publiques, Shorts, youtu.be
- **Twitch** : VODs (`twitch.tv/videos/...`), clips (`clips.twitch.tv/...` ou `twitch.tv/<channel>/clip/...`)
- **Upload direct** : fichier audio ou vidéo
- **Autres** : tout site supporté par [yt-dlp](https://github.com/yt-dlp/yt-dlp) (1300+)

## Usage

1. Colle une URL YouTube ou Twitch (ou upload un fichier audio/vidéo).
2. Le modèle Whisper `large-v3-turbo` est utilisé si un GPU NVIDIA est disponible ; sinon fallback CPU avec `small`.
3. Récupère le texte brut ou les sous-titres SRT.

## Limites

- Les Spaces HF gratuits tournent sur **CPU**. Pour un GPU, activez l'option GPU dans les paramètres du Space (payant après les crédits gratuits).
- Les vidéos YouTube doivent être publiques et accessibles depuis le serveur.
- Les retranscriptions ASR prennent ~1-5 min pour 30 min de contenu en CPU.

## Crédits

- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
