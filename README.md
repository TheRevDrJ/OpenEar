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

OpenEar taps audio directly from your soundboard and uses [Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (NVIDIA's fast, accurate speech recognition model) to transcribe sermons in real time. Captions stream to any phone, tablet, or display on your local Wi-Fi — in the original language or translated into any of 200+ languages and dialects using [NLLB-200](https://ai.meta.com/research/no-language-left-behind/) (Meta's "No Language Left Behind"), a high-quality local translation model.

No app to install. No account to create. Just open a browser.

Everything runs locally. No audio leaves your building. No internet connection is required during services.

## Features

- **Real-time captioning** — large, readable text streamed live to any device on the network
- **Real-time translation** — 200+ languages and dialects, any device, powered by Meta's NLLB-200 model
- **Server-side audio capture** — clean audio direct from the soundboard, not ambient room noise
- **Personal, not projected** — captions appear on individual devices; those who need them get them, without a distracting overlay for everyone else
- **Admin panel** — quality presets, settings, and controls for the sound booth operator
- **One-click installer** — `setup.bat` handles dependencies, models, and configuration on Windows
- **QR code access** — display a code in the lobby and anyone can connect instantly

## Why It Exists

Commercial captioning solutions like CaptionKit, Wordly, and Sunflower AI charge hundreds to thousands of dollars per year in subscriptions, require internet access during services, and route all sermon audio through third-party cloud servers. Human captioners run ~$150/hour.

OpenEar is different:

- **$0/month, $0/minute, forever.** Free and open source.
- **Runs 100% locally.** No internet required during services. No audio leaves your building.
- **One-time hardware cost — often $0.** Live captioning runs on an everyday Windows desktop — **no graphics card required.** Most churches already own a machine that qualifies. Adding translation into 200+ languages and dialects is an optional step that takes an **inexpensive NVIDIA graphics card — starting as low as $250.** Low-power models are slot-powered and drop into almost any PC with no power-supply upgrade.
- **No cloud dependency.** Your sermons stay in your building.

> *"We preach so that everyone can encounter God face to face. When technology can remove the glass, no one should have to pay for the window."*
> — Rev. Dr. Jonathan Mellette, Pastor & Developer

## Hardware Requirements

OpenEar grows with your needs. **Live captioning runs on hardware you almost certainly already have — no graphics card.** Translation into other languages is an optional step you can add later by dropping in a single inexpensive card.

### For live captioning (no GPU needed)

| Component | Spec |
|-----------|------|
| OS | Windows 10/11 (64-bit) |
| CPU | Any modern multi-core CPU (Ryzen 5 / Core i5, ~2018 or newer) |
| RAM | 8 GB |
| GPU | **None.** Transcription runs entirely on the CPU. |
| Storage | 5 GB available |

Captioning uses NVIDIA's Parakeet speech model, which runs on the CPU at roughly 34× real-time. No graphics card is involved at any point — this is the only way OpenEar does transcription, by design.

### To add translation (200+ languages)

Translation uses Meta's NLLB-200 model, which runs on an NVIDIA GPU. Add one card and the same live captions begin flowing in any of 200+ languages and dialects.

| Component | Spec |
|-----------|------|
| GPU | An **NVIDIA card with 6 GB+ VRAM** — starting as low as $250 |

> **The cheapest path:** OpenEar's translation needs only ~4.5 GB of VRAM, so an inexpensive NVIDIA card with 6 GB or more is plenty — these start around **$250 new** (prices vary). Look for a **low-power, slot-powered model**: it draws all its power from the PCIe slot itself, needs no supplemental power connector, and drops into almost any existing PC with **no power-supply upgrade.** For most churches, that one card is the entire hardware cost.

You also need a local network (church WiFi or a simple $30 router) for clients to connect.

### Experimental Platforms

**Mac Mini M4 (in testing):**
- Base model M4 Mac Mini (16GB unified memory, $599)
- Transcription runs on CPU via Parakeet — translation limited without NVIDIA GPU
- Status: v0.6 — actively being tested

### Reference Test Server

This is the actual hardware OpenEar is developed and tested on:

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 5700X |
| RAM | 64GB DDR4 |
| GPU | NVIDIA RTX 4090 24GB |
| Storage | 1TB NVMe M.2 |
| OS | Windows 11 |
| ASR model | Parakeet-TDT-0.6B (CPU) |
| Translation model | NLLB-200 3.3B INT8 (CUDA) |

**Observed resource usage while server is running:**

| Resource | Usage |
|----------|-------|
| RAM in use | ~7 GB of 64 GB |
| VRAM in use | ~4.4 GB of 24 GB |
| Disk space | ~100 GB (includes OS, models, and translation packs) |

*This server is significantly overpowered for OpenEar. The goal is to determine the minimum viable hardware — particularly whether the $599 Mac Mini M4 can deliver acceptable quality.*

## Quick Start (Windows)

### Prerequisites

1. **Python 3.13+** — install from the Microsoft Store (search "Python 3.13", click Get)
2. **NVIDIA GPU drivers — only if you're adding translation.** Captioning needs no GPU and no drivers. If you've installed an NVIDIA graphics card for translation, download the full **Game Ready** or **Studio** driver from [nvidia.com/drivers](https://www.nvidia.com/drivers). **Windows Update installs a basic display driver that does not include the CUDA runtime** — translation will fail silently if you rely on it. Even if `nvidia-smi` works, you may still be missing the CUDA runtime DLLs. Install the full driver from nvidia.com, restart, then run setup.

That's it. The setup script handles everything else.

### Install & Run

1. Clone or download the repo
2. Right-click `setup.bat` → **Run as administrator**
3. Wait for setup to complete (~15–30 minutes, downloads ~5.5GB of AI models)
4. Start the server: `openear.bat start`
5. Open `http://localhost/admin` to configure audio input
6. Congregation connects at `http://<server-ip>` on any device on the same WiFi
7. *(Optional)* Set a custom Windows desktop wallpaper — right-click desktop → Personalize

> **Note:** OpenEar runs on port 80 (standard HTTP) so users don't need to remember a port number. Your browser may show a "not secure" warning — this is normal. Everything runs on your local network. No data leaves the building.

## Roadmap

| Version | Feature | Status |
|---------|---------|--------|
| v0.1 | Proof of concept — live transcription via browser | ✓ Done |
| v0.2 | Server-side mic capture from soundboard | ✓ Done |
| v0.3 | UI polish, quality presets, admin settings | ✓ Done |
| v0.4 | Real-time translation — any language, any device | ✓ Done |
| v0.5 | One-click installer | ✓ Done |
| v0.6 | Mac Mini M4 development | ◐ In progress... |
| v0.7 | Testing at select churches | ◐ In progress... |
| v0.8 | Expanded testing & feedback | ○ Planned |
| v0.9 | Public release — manually deployable for any church | ○ Planned |
| v1.0 | Public release — automatically deployable for any church | ○ Planned |
| v1.1 | Bidirectional translation — any language input, any language output | ○ Planned |

## Get Involved

If you're a pastor interested in bringing OpenEar to your church, or a developer who wants to contribute, open an issue or reach out. This is a passion project — the goal is to remove barriers, not build a business.

## License

[AGPL-3.0](LICENSE) — use it, modify it, share it. Free forever. If you distribute or run a modified version as a network service, you must share your changes under the same license.

## Links

- **Website:** [openearproject.org](https://openearproject.org)
- **GitHub:** [TheRevDrJ/OpenEar](https://github.com/TheRevDrJ/OpenEar)

---

Built with [__Ephphatha__](https://github.com/TheRevDrJ). 🙌
