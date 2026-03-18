"""
rag/voice.py — Voice Chat Module
---------------------------------
Speech to Text  → ElevenLabs STT
Text to Speech  → ElevenLabs TTS
"""

import os
import logging
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

load_dotenv()
logging.basicConfig(level=logging.INFO)

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# ── Text to Speech ─────────────────────────────────────────────────────────────
def text_to_speech(text: str, voice_id: str = "JBFqnCBsd6RMkjVDRZzb") -> str:
    """
    Convert text to speech using ElevenLabs.
    Returns path to audio file.
    Default voice: George (natural, clear)
    """
    try:
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True
            )
        )

        # Save to temp file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        for chunk in audio:
            tmp_file.write(chunk)
        tmp_file.close()

        logging.info(f"🔊 TTS generated: {tmp_file.name}")
        return tmp_file.name

    except Exception as e:
        logging.error(f"❌ TTS failed: {e}")
        return None


# ── Speech to Text ─────────────────────────────────────────────────────────────
def record_audio(duration: int = 5, sample_rate: int = 16000) -> str:
    """
    Record audio from microphone.
    Returns path to recorded audio file.
    """
    try:
        logging.info(f"🎤 Recording for {duration} seconds...")
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()  # Wait until recording is done

        # Save to temp file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp_file.name, audio_data, sample_rate)
        tmp_file.close()

        logging.info(f"✅ Audio recorded: {tmp_file.name}")
        return tmp_file.name

    except Exception as e:
        logging.error(f"❌ Recording failed: {e}")
        return None


def speech_to_text(audio_path: str) -> str:
    """
    Convert speech to text using ElevenLabs STT.
    """
    try:
        with open(audio_path, "rb") as audio_file:
            result = client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1",
                language_code="en"
            )
        text = result.text.strip()
        logging.info(f"📝 STT result: {text[:60]}")
        return text

    except Exception as e:
        logging.error(f"❌ STT failed: {e}")
        return ""


def record_and_transcribe(duration: int = 5) -> str:
    """
    Record audio and transcribe to text.
    One function for easy use in UI.
    """
    audio_path = record_audio(duration)
    if not audio_path:
        return ""
    return speech_to_text(audio_path)


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔊 Testing TTS...")
    path = text_to_speech("Hello! I am your AI assistant. How can I help you today?")
    if path:
        print(f"✅ Audio saved: {path}")

    print("\n🎤 Testing STT — speak for 5 seconds...")
    text = record_and_transcribe(duration=5)
    print(f"✅ You said: {text}")
