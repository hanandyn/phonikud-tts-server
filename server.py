"""
Phonikud-TTS HTTP API Server (Renikud edition)
OpenAI-compatible /v1/audio/speech endpoint for Hebrew TTS
Uses Renikud G2P (higher accuracy, ~20MB) + Piper ONNX voices
"""
import io
import time
import logging
from pathlib import Path

import soundfile as sf
from flask import Flask, request, jsonify, send_file
from renikud_onnx import G2P
from piper_onnx import Piper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phonikud-tts")

app = Flask(__name__)

MODELS_DIR = Path("/app/models")

_renikud = None
_piper_shaul = None
_piper_michael = None

VOICE_MAP = {
    "shaul": "shaul.onnx",
    "michael": "michael.onnx",
}


def get_renikud():
    global _renikud
    if _renikud is None:
        path = MODELS_DIR / "renikud.onnx"
        log.info(f"Loading Renikud G2P model from {path}")
        _renikud = G2P(str(path))
    return _renikud


def get_piper(voice: str = "shaul"):
    global _piper_shaul, _piper_michael
    voice = voice.lower()
    if voice not in VOICE_MAP:
        voice = "shaul"
    if voice == "shaul":
        if _piper_shaul is None:
            _piper_shaul = Piper(
                str(MODELS_DIR / "shaul.onnx"),
                str(MODELS_DIR / "model.config.json"),
            )
        return _piper_shaul
    else:
        if _piper_michael is None:
            _piper_michael = Piper(
                str(MODELS_DIR / "michael.onnx"),
                str(MODELS_DIR / "model.config.json"),
            )
        return _piper_michael


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "g2p": "renikud", "voices": list(VOICE_MAP.keys())})


@app.route("/v1/audio/speech", methods=["POST"])
def speech():
    data = request.get_json(silent=True) or {}
    text = data.get("input", "").strip()
    voice = data.get("voice", "shaul")
    response_format = data.get("response_format", "wav")
    speed = float(data.get("speed", 1.0))

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if voice not in VOICE_MAP:
        return jsonify({"error": f"Unknown voice '{voice}'. Available: {list(VOICE_MAP.keys())}"}), 400

    log.info(f"TTS: voice={voice} text_len={len(text)} fmt={response_format}")

    try:
        start = time.time()

        # Renikud: one-step text → phonemes (no diacritics step needed!)
        renikud = get_renikud()
        phonemes = renikud.phonemize(text)

        # Piper: phonemes → audio
        piper = get_piper(voice)
        if speed != 1.0:
            samples, sample_rate = piper.create(
                phonemes, is_phonemes=True, length_scale=1.0 / speed
            )
        else:
            samples, sample_rate = piper.create(phonemes, is_phonemes=True)

        elapsed = time.time() - start
        duration = len(samples) / sample_rate
        rtf = elapsed / duration if duration > 0 else 0
        log.info(f"Generated {duration:.1f}s audio in {elapsed:.1f}s (RTF: {rtf:.2f}x)")

        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV")
        buf.seek(0)

        return send_file(
            buf,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=f"speech.{response_format}",
        )

    except Exception as e:
        log.exception("TTS generation failed")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/voices", methods=["GET"])
def voices():
    return jsonify({
        "voices": [
            {"id": "shaul", "name": "Shaul", "language": "he", "gender": "male"},
            {"id": "michael", "name": "Michael", "language": "he", "gender": "male"},
        ]
    })


if __name__ == "__main__":
    log.info("Starting phonikud-tts server (Renikud edition) on port 8880")
    get_renikud()
    get_piper("shaul")
    log.info("Models loaded. Ready.")
    app.run(host="0.0.0.0", port=8880)
