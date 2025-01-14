import pyaudio

p = pyaudio.PyAudio()

print("Verfügbare Audio-Eingabegeräte:")
info = p.get_host_api_info_by_index(0)  # Verwende den Standard-Host-API-Index (normalerweise 0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        devinfo = p.get_device_info_by_index(i)
        print(f"Index: {i}, Name: {devinfo.get('name')}")

p.terminate()

