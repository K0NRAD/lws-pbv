import logging
import os
import queue
import threading
import warnings
import wave
from datetime import datetime

import pyaudio
import torch
import webrtcvad
import whisper

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("torch").setLevel(logging.ERROR)

SAMPLE_RATE = 16000
FRAME_DURATION = 30  # in Milliseconds
CHUNK = int(SAMPLE_RATE * FRAME_DURATION / 1000)


RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")

if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)


class AudioRecorder:
    def __init__(self):
        self.CHUNK = CHUNK
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = SAMPLE_RATE
        self.INPUT_DEVICE_INDEX = 1
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.recording_thread = None
        self.vad = webrtcvad.Vad(2)
        self.audio_queue = queue.Queue()

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def record(self):
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=self.INPUT_DEVICE_INDEX,
            stream_callback=self.audio_callback
        )

        self.stream.start_stream()

        while self.is_recording:
            try:
                data = self.audio_queue.get(timeout=1.0)
                self.frames.append(data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Fehler bei der Aufnahme: {e}")
                break

        self.stream.stop_stream()
        self.stream.close()

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            self.audio_queue = queue.Queue()
            self.recording_thread = threading.Thread(target=self.record)
            self.recording_thread.start()
            print("Aufnahme gestartet...")

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join()
            print("Aufnahme beendet.")

    def save_audio(self, filename):
        if not self.frames:
            print("Keine Audiodaten zum Speichern vorhanden.")
            return False

        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            return True
        except Exception as e:
            print(f"Fehler beim Speichern der Audiodatei: {e}")
            return False

    def close(self):
        self.p.terminate()


def transcribe_audio(audio_file):
    try:
        print("Lade Whisper Modell...")

        # Monkey patch torch.load to use weights_only=True
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=True)
        model = whisper.load_model("small", device="cpu")

        # Restore original torch.load
        torch.load = original_load
        print("Transkribiere Audio...")
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        print(f"Fehler bei der Transkription: {e}")
        return None


def main():
    print("Audio Aufnahme und Transkription mit Sprechpausenerkennung")
    print("=" * 60)
    recorder = None

    try:
        recorder = AudioRecorder()
        silence_threshold_seconds = 1.0
        frames_per_second = int(recorder.RATE / recorder.CHUNK)
        silence_frames_threshold = int(silence_threshold_seconds * frames_per_second)
        silence_frames = 0
        is_speaking = False

        stream = recorder.p.open(
            format=recorder.FORMAT,
            channels=recorder.CHANNELS,
            rate=recorder.RATE,
            input=True,
            frames_per_buffer=recorder.CHUNK,
            input_device_index=recorder.INPUT_DEVICE_INDEX,
        )

        print("Warte auf Sprache...")

        while True:
            try:
                data = stream.read(recorder.CHUNK, exception_on_overflow=False)
                is_speech = recorder.vad.is_speech(data, recorder.RATE)

                if is_speech and not is_speaking:
                    is_speaking = True
                    silence_frames = 0
                    recorder.start_recording()
                elif is_speaking and not is_speech:
                    silence_frames += 1
                    if silence_frames >= silence_frames_threshold:
                        is_speaking = False
                        recorder.stop_recording()

                        if recorder.frames:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            audio_file = os.path.join(RECORDINGS_DIR, f"audio_{timestamp}.wav")

                            if recorder.save_audio(audio_file):
                                print(f"Audio gespeichert in: {audio_file}")
                                transcription = transcribe_audio(audio_file)
                                if transcription:
                                    text_file = os.path.join(RECORDINGS_DIR, f"transcript_{timestamp}.txt")
                                    with open(text_file, 'w', encoding='utf-8') as f:
                                        f.write(transcription)
                                    print("\nTranskription:")
                                    print("-" * 50)
                                    print(transcription)
                                    print("-" * 50)
                                    print(f"\nTranskription gespeichert in: {text_file}")
                                    print("\nWarte auf Sprache...")
                            else:
                                print("Fehler beim Speichern der Audiodatei.")
                                print("\nWarte auf Sprache...")

            except IOError as e:
                if e.errno == -9981:  # Input overflow
                    continue  # Skip this chunk and continue
                else:
                    raise

    except KeyboardInterrupt:
        print("\nProgramm wurde durch Benutzer beendet.")
    finally:
        if recorder:
            recorder.close()
            stream.stop_stream()
            stream.close()
            print("PyAudio geschlossen.")


if __name__ == "__main__":
    main()
