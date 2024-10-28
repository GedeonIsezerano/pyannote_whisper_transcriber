from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import yaml
import torch
import numpy as np
import gc
from importlib import resources
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
from tempfile import NamedTemporaryFile
from pydantic import BaseModel, Field



class TranscriptionSegment(BaseModel):
    start_time: str = Field(..., description="Start time of the segment in HH:MM:SS format")
    end_time: str = Field(..., description="End time of the segment in HH:MM:SS format")
    speaker: str = Field(..., description="Speaker identifier")
    text: str = Field(..., description="Transcribed text")

    def __str__(self) -> str:
        return f'[ {self.start_time} -- {self.end_time} ] {self.speaker} : {self.text}'


class AudioTranscriber:
    def __init__(self, pyannote_config_yaml: str = "pyannote_diarization_config.yaml", 
                 whisper_model_name: str = "turbo",
                 device: str = "cuda"):
        """Initialize the AudioTranscriber with specified models and device.

        Args:
            pyannote_config_yaml: Name of the PyAnnote configuration file
            whisper_model_name: Name of the Whisper model to use
            device: Device to run models on ("cuda" or "cpu")
        """
        self.device = torch.device(device) if device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
        self.config = self._load_config(pyannote_config_yaml)
        self.whisper_model_name = whisper_model_name
        
        # Initialize components as None
        self.pipeline: Optional[Pipeline] = None
        self.whisper = None
        self.audio: Optional[AudioSegment] = None
        self.diarization_result: Optional[List[str]] = None

    def _load_config(self, config_filename: str) -> dict:
        """Load and prepare the PyAnnote configuration."""
        config_path = resources.files("transcriber.models").joinpath(config_filename)
        model_dir = Path(config_path).parent

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update paths to absolute
        config['pipeline']['params']['embedding'] = str(model_dir / config['pipeline']['params']['embedding'])
        config['pipeline']['params']['segmentation'] = str(model_dir / config['pipeline']['params']['segmentation'])
        
        return config

    def initialize(self) -> None:
        """Initialize the PyAnnote pipeline and Whisper model."""
        if self.pipeline is not None:
            return

        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(self.config, tmp_file)
            tmp_config_path = tmp_file.name

        try:
            self.pipeline = Pipeline.from_pretrained(tmp_config_path)
            self.pipeline.to(self.device)
            self.whisper = whisper.load_model(self.whisper_model_name)
        finally:
            Path(tmp_config_path).unlink()
    @staticmethod
    def _read_audio_segment(audio_segment: AudioSegment) -> np.ndarray:
        """Convert AudioSegment to numpy array for Whisper."""
        return np.frombuffer(audio_segment.raw_data, np.int16).astype(np.float32) / 32768.0

    @staticmethod
    def _parse_timestamp(time_str: str) -> int:
        """Convert timestamp string to milliseconds."""
        hours, minutes, seconds = map(float, time_str.split(":"))
        return int((hours * 3600 + minutes * 60 + seconds) * 1000)

    def _perform_diarization(self, audio_file: str) -> List[str]:
        """Perform speaker diarization on the audio file."""
        result = str(self.pipeline(audio_file)).split('\n')
        print(f"Diarization result: {result}")
        return result

    def _process_segment(self, segment: str) -> TranscriptionSegment:
        """Process a single diarization segment and return a TranscriptionSegment."""
        parts = segment.split(" ")
        start_time = parts[1]
        end_time = parts[4][:-1]
        speaker = parts[6]
        print(f"Processing segment: {start_time} - {end_time} - {speaker}")
        start_ms = self._parse_timestamp(start_time)
        end_ms = self._parse_timestamp(end_time)

        audio_segment = self._read_audio_segment(self.audio[start_ms:end_ms])
        
        transcription = self.whisper.transcribe(audio_segment, fp16=False if str(self.device) == "cpu" else True)

        return TranscriptionSegment(
            start_time=start_time,
            end_time=end_time,
            speaker=speaker,
            text=transcription["text"]
        )

    def transcribe(self, audio_file: str, output_file: str) -> None:
        """
        Transcribe an audio file with speaker diarization.

        Args:
            audio_file: Path to the input audio file
            output_file: Path to save the transcription
        """
        self.initialize()
        
        # Load audio
        self.audio = AudioSegment.from_mp3(audio_file).set_frame_rate(16000)
        
        # Perform diarization
        self.diarization_result = self._perform_diarization(audio_file)
        if not self.diarization_result:
            raise ValueError("Diarization failed. Please check the audio file.")

        # Process segments and write output
        with open(output_file, "w", encoding="utf-8") as f:
            for segment in self.diarization_result:
                if not segment.strip():  # Skip empty lines
                    continue
                    
                transcription_segment = self._process_segment(segment)
                f.write(f"{transcription_segment}\n")
                gc.collect()

        # Cleanup
        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up resources to free memory."""
        del self.audio
        del self.diarization_result
        self.audio = None
        self.diarization_result = None
        gc.collect()

# Usage example:
# transcriber = AudioTranscriber()
# transcriber.transcribe("audio.mp3", "transcription_output.txt")
