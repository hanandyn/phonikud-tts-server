"""
Phonikud-TTS HTTP API Server (Renikud edition)
OpenAI-compatible /v1/audio/speech endpoint for Hebrew TTS
Voices: shaul (Piper male), michael (Piper male), maia (StyleTTS2 female)
"""
import io
import time
import logging
from pathlib import Path
from functools import wraps
import os
import secrets

import soundfile as sf
from flask import Flask, request, jsonify, send_file
from renikud_onnx import G2P
from piper_onnx import Piper
from style_onnx import StyleTTS2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phonikud-tts")

app = Flask(__name__)

# --- Auth ---
API_USERNAME = os.environ.get("API_USERNAME", "")
API_PASSWORD = os.environ.get("API_PASSWORD", "")
REQUIRE_AUTH = bool(API_USERNAME and API_PASSWORD)

def check_auth(username, password):
    return secrets.compare_digest(username, API_USERNAME) and secrets.compare_digest(password, API_PASSWORD)

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not REQUIRE_AUTH:
            return f(*args, **kwargs)
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return jsonify({"error": "Unauthorized"}), 401, {"WWW-Authenticate": 'Basic realm="TTS"'}
        return f(*args, **kwargs)
    return decorated

MODELS_DIR = Path("/app/models")

_renikud = None
_piper = {}
_styletss2 = None
_female_style = None

VOICE_MAP = {
    "shaul": {"engine": "piper", "model": "shaul.onnx"},
    "michael": {"engine": "piper", "model": "michael.onnx"},
    "maia": {"engine": "styletss2", "style": "636_female"},
    "maia-alt": {"engine": "styletss2", "style": "female1"},
}


def get_renikud():
    global _renikud
    if _renikud is None:
        _renikud = G2P(str(MODELS_DIR / "renikud.onnx"))
        log.info("Renikud G2P loaded")
    return _renikud


def get_piper(voice: str):
    global _piper
    model_file = VOICE_MAP[voice]["model"]
    if model_file not in _piper:
        _piper[model_file] = Piper(
            str(MODELS_DIR / model_file),
            str(MODELS_DIR / "model.config.json"),
        )
        log.info(f"Piper voice '{voice}' loaded")
    return _piper[model_file]


def get_styletss2(style_name: str):
    global _styletss2, _female_style
    style_file = f"{style_name}_style.npy"
    if _styletss2 is None or _female_style != style_name:
        _styletss2 = StyleTTS2(
            str(MODELS_DIR / "libritts_hebrew.onnx"),
            str(MODELS_DIR / style_file),
        )
        _female_style = style_name
        log.info(f"StyleTTS2 voice '{style_name}' loaded")
    return _styletss2


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "g2p": "renikud",
        "voices": list(VOICE_MAP.keys()),
        "auth_enabled": REQUIRE_AUTH,
    })


@app.route("/v1/audio/speech", methods=["POST"])
@auth_required
def speech():
    data = request.get_json(silent=True) or {}
    text = data.get("input", "").strip()
    voice = data.get("voice", "maia")
    response_format = data.get("response_format", "wav")
    speed = float(data.get("speed", 1.0))

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if voice not in VOICE_MAP:
        return jsonify({
            "error": f"Unknown voice '{voice}'. Available: {list(VOICE_MAP.keys())}"
        }), 400

    voice_cfg = VOICE_MAP[voice]
    log.info(f"TTS: voice={voice} ({voice_cfg['engine']}) text_len={len(text)}")

    try:
        start = time.time()

        renikud = get_renikud()
        phonemes = renikud.phonemize(text)

        if voice_cfg["engine"] == "piper":
            piper = get_piper(voice)
            if speed != 1.0:
                samples, sample_rate = piper.create(
                    phonemes, is_phonemes=True, length_scale=1.0 / speed
                )
            else:
                samples, sample_rate = piper.create(phonemes, is_phonemes=True)
        else:
            tts = get_styletss2(voice_cfg["style"])
            samples, sample_rate = tts.create(phonemes, speed=speed)

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
@auth_required
def voices():
    return jsonify({
        "voices": [
            {"id": "maia", "name": "mAIa", "language": "he", "gender": "female"},
            {"id": "maia-alt", "name": "mAIa (alt)", "language": "he", "gender": "female"},
            {"id": "shaul", "name": "Shaul", "language": "he", "gender": "male"},
            {"id": "michael", "name": "Michael", "language": "he", "gender": "male"},
        ]
    })


if __name__ == "__main__":
    log.info("Starting phonikud-tts server (Renikud edition) on port 8880")
    if REQUIRE_AUTH:
        log.info("Auth enabled - protected endpoints require credentials")
    else:
        log.warning("No API_USERNAME/API_PASSWORD set - endpoints are OPEN")
    get_renikud()
    get_piper("shaul")
    log.info("Models loaded. Ready.")
    app.run(host="0.0.0.0", port=8880)
