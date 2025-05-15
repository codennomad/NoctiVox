# 🧠 NoctiVox

**AI-powered video pipeline for transcription, intelligent titling, and content automation.**

NoctiVox is a modular tool that listens to one or multiple videos, generates an accurate transcription using Whisper, sends that text to an LLM via OpenRouter, and renames the video file with a highly relevant, SEO-friendly title — ready for publishing.

---

## 🚀 Features

- 🎙️ Transcribe audio with Whisper (via `faster-whisper` or `openai-whisper`)
- 🧠 Generate smart video titles using GPT (via [OpenRouter](https://openrouter.ai))
- 📝 Save transcripts in readable, subtitle-friendly format
- 🪪 Rename original video file based on generated title
- 🧾 Log everything using `loguru`
- 🎯 Support for multiple videos in batch mode

---

## 📈 Vision: What's Coming Next

> NoctiVox is not just a transcription tool. It's becoming a full-on **content automation assistant** for creators and analysts.

- 📅 Auto-post to **YouTube**, **Instagram**, **TikTok**
- 🧠 Choose best **publishing time and SEO** using AI
- 🏷️ Auto-generate tags, descriptions, and thumbnails
- 🌍 Multilingual transcription and translation
- 🎬 Subtitle export in `.srt`, `.vtt`, `.ass`

---

## 🧑‍💻 Usage

```bash
# Activate virtual environment
source noctivox_env/bin/activate  # or .\noctivox_env\Scripts\activate on Windows
```

# Install dependencies
```bash
pip install -r requirements.txt
```


## 🔐 .env Configuration

#Your .env file should include the following:

```bash
OPENROUTER_API_KEY=your_openrouter_key_here
HF_TOKEN=your_huggingface_token_here
```

These are required to use external APIs like OpenRouter (for GPT-based title generation) and Hugging Face models (if integrated in the future).


## 📦 Project Structure

```bash 

📁 NoctiVox/
├── NoctiVox.py                # Main logic
├── STT.py                     # Transcription (Whisper + diarization)
├── LLM.py                     # Title generation with GPT
├── requirements.txt
├── logs/noctivox.log
├── .env
└── output/
    ├── *_transcription.txt
    └── processing_report.txt
```

## 🙋‍♂️ Author

Gabriel Henrique Ferreira Vieira
🔗 [LinkedIn](https://www.linkedin.com/in/gabrielhenriquefv/)
📧 gabrielheh03@gmail.com
🧠 AI Intern Candidate | NeuroTech | Python Developer

## 📄 License

MIT License
