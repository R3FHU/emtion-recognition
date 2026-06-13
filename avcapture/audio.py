import pyaudio
import time
from collections import deque

class AudioCapture:
    def __init__(self, rate=16000, chunk=1024,start_time=None):
        self.rate = rate
        self.chunk = chunk
        self.start_time = start_time or time.perf_counter()

        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        self.buffer = deque(maxlen=100)
        self.running = False
        self.start_time=time.perf_counter()
        self.total_samples=0

    def start(self):
        import threading
        self.running = True
        self.thread = threading.Thread(target=self._record)
        self.thread.daemon = True
        self.thread.start()

    def _record(self):
        while self.running:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            timestamp = self.total_samples / float(self.rate)
            self.buffer.append((timestamp, data))
            self.total_samples += self.chunk

    def get_audio_segment(self, current_time, window=0.5):
        """取某一时间点前window秒的音频"""
        result = []

        for t, data in self.buffer:
            if abs(t - current_time) <= window:
                result.append(data)

        return result

    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()