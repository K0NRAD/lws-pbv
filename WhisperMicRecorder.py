import queue
import signal
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pyaudio
import whisper
from scipy import signal as scipy_signal


class WhisperMicRecorder:
    def __init__(self, model_size='medium', device_index=None):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.recording = False
        self.device_index = device_index

        self.energy_threshold = 300
        self.dynamic_energy = True
        self.noise_reduce_amount = 0.75

        print(f"Lade Whisper Modell ({model_size})...")
        self.model = whisper.load_model(model_size)
        print("Modell geladen!")

        self.whisper_options = {
            "language": "de",
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.2,
            "condition_on_previous_text": True,
            "initial_prompt": "Dies ist eine Aufnahme in deutscher Sprache."
        }

        self.audio_queue = queue.Queue()
        self.p = pyaudio.PyAudio()

        self.noise_profile = None
        self.noise_samples = []

        signal.signal(signal.SIGINT, self.signal_handler)

    def enhance_audio(self, audio_data):
        try:
            # Konvertiere zu numpy array und erstelle eine Kopie
            data = np.frombuffer(audio_data, dtype=np.float32).copy()

            # Hochpassfilter
            b, a = scipy_signal.butter(4, 80 / (self.RATE / 2), 'highpass')
            filtered_data = scipy_signal.filtfilt(b, a, data)

            # Normalisierung (nur wenn Audio nicht zu leise)
            max_value = np.max(np.abs(filtered_data))
            if max_value > 0.01:
                filtered_data = filtered_data / max_value * 0.9

            return filtered_data.astype(np.float32)

        except Exception as e:
            print(f"Fehler bei der Audioverarbeitung: {e}")
            return np.frombuffer(audio_data, dtype=np.float32)

    def record_audio(self):
        """Nimmt Audio auf und verarbeitet es"""
        try:
            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.CHUNK
            )

            print("\nAufnahme läuft... (Drücke Ctrl+C zum Beenden)")

            while self.recording:
                try:
                    audio_chunk = stream.read(self.CHUNK, exception_on_overflow=False)

                    enhanced_audio = self.enhance_audio(audio_chunk)

                    self.audio_queue.put(enhanced_audio.tobytes())

                except Exception as e:
                    print(f"Fehler bei der Aufnahme: {e}")
                    continue

            stream.stop_stream()
            stream.close()

        except Exception as e:
            print(f"Fehler beim Öffnen des Audio-Streams: {e}")

    def process_audio(self):
        audio_buffer = []
        last_process_time = time.time()

        while self.recording:
            current_time = time.time()

            while not self.audio_queue.empty():
                audio_buffer.append(self.audio_queue.get())

            if current_time - last_process_time >= 2.0 and audio_buffer:
                try:
                    audio_data = np.frombuffer(b''.join(audio_buffer), dtype=np.float32)

                    result = self.model.transcribe(
                        audio_data,
                        **self.whisper_options
                    )

                    if result['text'].strip():
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"\n[{timestamp}] {result['text'].strip()}")

                except Exception as e:
                    print(f"Fehler bei der Transkription: {e}")

                # Buffer leeren und Zeit zurücksetzen
                audio_buffer = []
                last_process_time = current_time

            time.sleep(0.1)

    def start_recording(self):
        self.recording = True

        # Threads starten
        record_thread = threading.Thread(target=self.record_audio)
        process_thread = threading.Thread(target=self.process_audio)

        record_thread.start()
        process_thread.start()

        try:
            while self.recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nBeende Aufnahme...")
            self.recording = False

        # Warte auf Thread-Ende
        record_thread.join()
        process_thread.join()

    def signal_handler(self, signum, frame):
        print("\nBeende Aufnahme...")
        self.recording = False
        time.sleep(1)
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        try:
            self.p.terminate()
        except:
            pass


def main():
    recorder = WhisperMicRecorder(
        model_size='medium'
    )

    try:
        recorder.start_recording()
    except KeyboardInterrupt:
        print("\nProgramm wurde beendet")
    finally:
        recorder.cleanup()


if __name__ == "__main__":
    main()
