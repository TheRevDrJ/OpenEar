<p align="center">
  <img src="static/android-chrome-512x512.png" alt="OpenEar Logo" width="200">
</p>

<h1 align="center">OpenEar</h1>

<p align="center">
  <strong>Free, open-source, real-time AI captioning and translation for churches.</strong><br>
  No cloud. No subscription. No data leaves the building.
</p>

---

## What It Does

A small GPU-powered server runs [Whisper AI](https://github.com/SYSTRAN/faster-whisper) on the local network. Anyone on the church WiFi opens a URL on their phone or tablet and sees the sermon transcribed in real time as large, readable text. No app install. No account. Just a URL.

Built to make worship accessible.

## Why It Exists

Commercial captioning solutions like CaptionKit, Wordly, and Sunflower AI charge hundreds to thousands of dollars per year in subscriptions, require internet access during services, and route all sermon audio through third-party cloud servers. Human captioners run ~$150/hour.

OpenEar is different:

- **$0/month, $0/minute, forever.** Free and open source.
- **Runs 100% locally.** No internet required during services. No audio leaves your building.
- **One-time hardware cost.** A ~$600 Mac Mini or any PC with an NVIDIA GPU.
- **Personal, not projected.** Captions appear on individual devices — those who need them get them, without a distracting overlay for everyone else.
- **Real-time translation on the roadmap.** One sermon, every language, every phone.

## Requirements

- A PC with an NVIDIA GPU (8GB+ VRAM recommended — a 3060 12GB works great)
- Windows 10/11 with Python 3.13+
- A local network (church WiFi)

## Quick Start
```
pip install --upgrade pip
pip install faster-whisper fastapi uvicorn[standard] python-multipart
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
cd OpenEar
python server.py
```

Open `http://<server-ip>:8620` on any device on the same network.

## Prerequisites

1. Python 3.13 from the Microsoft Store
2. NVIDIA GPU driver
3. CUDA Toolkit (`winget install Nvidia.CUDA`)
4. FFmpeg (`winget install Gyan.FFmpeg`)

## Roadmap

| Version | Feature | Status |
|---------|---------|--------|
| v0.1 | Proof of concept — live transcription via browser | ✓ Done |
| v0.2 | Server-side mic capture from soundboard | ✓ Done |
| v0.3 | UI polish, quality presets, admin settings | ✓ Done |
| v0.4 | Real-time translation — any language, any device | ✓ Done |
| v0.5 | One-click installer | ✓ Done |
| v0.6 | Mac Mini testing | ◐ In progress... |
| v0.7 | Testing at select churches | ○ Planned |
| v0.8 | Expanded testing & feedback | ○ Planned |
| v0.9 | Public release — manually deployable for any church | ○ Planned |
| v1.0 | Public release — automatically deployable for any church | ○ Planned |

## Origin

The name OpenEar comes from the spirit of open-source accessibility. The project was inspired by Mark 7:34 — *Ephphatha*, "Be opened."

Built at Oak Park United Methodist Church in Temple, TX, because every congregation deserves accessibility — not just the ones that can afford it.

## Get Involved

If you're a pastor interested in bringing OpenEar to your church, or a developer who wants to contribute, open an issue or reach out. This is a passion project — the goal is to remove barriers, not build a business.

## License

AGPL-3.0 — use it, modify it, share it. Free forever. If you distribute or run a modified version, you must share your changes under the same license.
