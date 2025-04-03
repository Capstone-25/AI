# test_record.py

import sounddevice as sd
from scipy.io.wavfile import write
import os

def record_audio_to(filename="test.wav", duration=5, fs=16000):
    print("🎙️ 5초간 말해주세요... (녹음 중)")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print(f"✅ 녹음 완료: {filename}")

    # 저장된 파일 바로 재생 (Windows 전용)
    os.system(f"start {filename}")

if __name__ == "__main__":
    record_audio_to()
