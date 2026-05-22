"""
Phonikud-TTS HTTP API Server
Provides OpenAI-compatible /v1/audio/speech endpoint for Hebrew TTS
"""
import io
import time
import logging
from pathlib import Path
from typing import Optional

import soundfile as sf
from flask import Flask, request, jsonify, send_file

from phonikud_tts import Phonikud, phonemize, Piper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phonikud-tts")

app = Flask(__name__)

MODELS_DIR = Path("/app/models")

# Lazy-loaded models
_phonikud = None
_piper_shaul = None
_piper_michael = None

VOICE_MAP = {
    "shaul": "shaul.onnx",
    "michael": "michael.onnx",
}


def get_phonikud():
    global _phonikud
    if _phonikud is None:
        path = MODELS_DIR / "phonikud-1.0.int8.onnx"
        log.info(f"Loading Phonikud G2P model from {path}")
        _phonikud = Phonikud(str(path))
    return _phonikud


def get_piper(voice: str = "shaul"):
    global _piper_shaul, _piper_michael
    voice = voice.lower()
    if voice not in VOICE_MAP:
        voice = "shaul"

    if voice == "shaul":
        if _piper_shaul is None:
            model_path = MODELS_DIR / "shaul.onnx"
            config_path = MODELS_DIR / "model.config.json"
            log.info(f"Loading Piper Shaul voice from {model_path}")
            _piper_shaul = Piper(str(model_path), str(config_path))
        return _piper_shaul
    else:
        if _piper_michael is None:
            model_path = MODELS_DIR / "michael.onnx"
            config_path = MODELS_DIR / "model.config.json"
            log.info(f"Loading Piper Michael voice from {model_path}")
            _piper_michael = Piper(str(model_path), str(config_path))
        return _piper_michael


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "voices": list(VOICE_MAP.keys())})


@app.route("/v1/audio/speech", methods=["POST"])
def speech():
    """OpenAI-compatible TTS endpoint"""
    data = request.get_json(silent=True) or {}

    text = data.get("input", "").strip()
    voice = data.get("voice", "shaul")
    response_format = data.get("response_format", "wav")
    speed = float(data.get("speed", 1.0))

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Validate voice
    if voice not in VOICE_MAP:
        return jsonify({
            "error": f"Unknown voice '{voice}'. Available: {list(VOICE_MAP.keys())}"
        }), 400

    log.info(f"TTS request: voice={voice}, text_len={len(text)}, format={response_format}")

    try:
        start = time.time()

        # Step 1: Add diacritics (nikud)
        phonikud = get_phonikud()
        vocalized = phonikud.add_diacritics(text)

        # Step 2: Convert to phonemes
        phonemes = phonemize(vocalized, schema="plain")

        # Step 3: Generate audio with Piper
        piper = get_piper(voice)
        if speed != 1.0:
            # Piper supports length_scale for speed adjustment
            samples, sample_rate = piper.create(
                phonemes,
                is_phonemes=True,
                length_scale=1.0 / speed
            )
        else:
            samples, sample_rate = piper.create(phonemes, is_phonemes=True)

        elapsed = time.time() - start
        duration = len(samples) / sample_rate
        rtf = elapsed / duration if duration > 0 else 0
        log.info(f"Generated {duration:.1f}s audio in {elapsed:.1f}s (RTF: {rtf:.2f}x)")

        # Step 4: Encode to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV")
        buf.seek(0)

        mimetype = "audio/wav"
        if response_format == "mp3":
            mimetype = "audio/mpeg"
        elif response_format == "opus":
            mimetype = "audio/opus"

        return send_file(
            buf,
            mimetype=mimetype,
            as_attachment=True,
            download_name=f"speech.{response_format}"
        )

    except Exception as e:
        log.exception("TTS generation failed")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/voices", methods=["GET"])
def voices():
    """List available voices"""
    return jsonify({
        "voices": [
            {"id": "shaul", "name": "Shaul", "language": "he", "gender": "male"},
            {"id": "michael", "name": "Michael", "language": "he", "gender": "male"},
        ]
    })


if __name__ == "__main__":
    log.info("Starting phonikud-tts server on port 8880")
    # Preload models at startup
    get_phonikud()
    get_piper("shaul")
    log.info("Models loaded. Ready.")
    app.run(host="0.0.0.0", port=8880)
