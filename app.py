import pyaudio
import wave
import numpy as np
import whisper
import os
import threading
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")

if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

class AudioRecorder:
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.recording_thread = None

    def record(self):
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        self.frames = []
        while self.is_recording:
            try:
                data = self.stream.read(self.CHUNK)
                self.frames.append(data)
            except Exception as e:
                print(f"Fehler bei der Aufnahme: {e}")
                break

        self.stream.stop_stream()
        self.stream.close()

    def start_recording(self):
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self.record)
        self.recording_thread.start()
        print("Aufnahme läuft... Drücke Enter zum Beenden.")

    def stop_recording(self):
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()

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
        model = whisper.load_model("small")
        print("Transkribiere Audio...")
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        print(f"Fehler bei der Transkription: {e}")
        return None


def main():
    print("Audio Aufnahme und Transkription")
    print("=" * 40)
    print("Dieses Programm nimmt Audio auf und transkribiert es mit Whisper.")
    print(f"Aufnahmen werden gespeichert in: {RECORDINGS_DIR}")
    print("=" * 40 + "\n")

    while True:
        recorder = None
        try:
            recorder = AudioRecorder()

            print("\nBereit für neue Aufnahme. Drücken Sie Enter zum Starten...")
            input()

            recorder.start_recording()

            input()

            print("\nBeende Aufnahme...")
            recorder.stop_recording()

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

        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")

        finally:
            if recorder:
                recorder.close()

        if input("\nMöchtest Du eine weitere Aufnahme machen? (j/n): ").lower() != 'j':
            break

    print("Programm beendet.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramm wurde durch Benutzer beendet.")
    except Exception as e:
        print(f"\nUnerwarteter Fehler: {e}")
    finally:
        print("Programm wird beendet...")
