import webrtcvad
import pyaudio
import struct

# VAD-Objekt erstellen und Aggressivität einstellen (0-3)
vad = webrtcvad.Vad(2)

# Audioparameter festlegen
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # in Millisekunden
CHUNK = int(SAMPLE_RATE * FRAME_DURATION / 1000)

# PyAudio-Stream initialisieren
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Spracherkennung gestartet. Drücken Sie Strg+C zum Beenden.")

try:
    while True:
        frame = stream.read(CHUNK)
        is_speech = vad.is_speech(frame, SAMPLE_RATE)

        if is_speech:
            print("Sprache erkannt")
        else:
            print("Sprechpause")

except KeyboardInterrupt:
    print("Programm beendet.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
