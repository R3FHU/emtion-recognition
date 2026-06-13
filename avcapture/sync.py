class AVSynchronizer:
    def __init__(self, audio_capture):
        self.audio_capture = audio_capture

    def get_synced_data(self, frame, timestamp):
        audio_segment = self.audio_capture.get_audio_segment(timestamp)

        return {
            "frame": frame,
            "timestamp": timestamp,
            "audio": audio_segment
        }