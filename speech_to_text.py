import threading

import sounddevice as sd

import numpy as np

import whisper


# Shared state for recording
recording_data = []
is_recording = False
SAMPLERATE = 16000
CHANNELS = 1

# Lock for thread-safe operations
recording_lock = threading.Lock()

# Whisper model loading (only once to save time)
model = whisper.load_model("small", in_memory=True)

# Audio stream reference
audio_stream = None


# Callback function for audio recording
def audio_callback_(indata, frames, time, status):
    if status:
        print("Audio input status:", status)
    with recording_lock:
        if is_recording:
            recording_data.append(indata.copy())


def start_recording_():
    global is_recording, recording_data, audio_stream

    if is_recording:
        return False

    recording_data = []  # Reset data buffer
    is_recording = True

    # Open the audio stream
    audio_stream = sd.InputStream(
        callback=audio_callback_, channels=CHANNELS, samplerate=SAMPLERATE
    )
    audio_stream.start()

    return True


def stop_recording_():
    global is_recording, audio_stream

    if not is_recording:
        return None

    is_recording = False

    # Stop and close the audio stream
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()

    audio_data = (
        np.concatenate(recording_data, axis=0).flatten()
        if recording_data
        else np.array([])
    )

    if len(audio_data) == 0:
        return None

    result = model.transcribe(
        audio_data, fp16=False, initial_prompt="The text is in English."
    )

    if not result["segments"]:
        return None

    avg_logprobs = [segment["avg_logprob"] for segment in result["segments"]]
    if sum(avg_logprobs) / len(avg_logprobs) < -3:
        return None

    no_speech_probs = [segment["no_speech_prob"] for segment in result["segments"]]
    if sum(no_speech_probs) / len(no_speech_probs) > 0.5:
        return None

    return result
