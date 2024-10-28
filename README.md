# Audio Transcriber with Speaker Diarization

A Python tool that combines OpenAI's Whisper and Pyannote.audio to transcribe audio files and identify different speakers. Perfect for meetings, interviews, or any multi-speaker audio content.


## Features
- 🎯 Accurate audio transcription using OpenAI's Whisper
- 👥 Speaker identification with Pyannote.audio
- ⏱️ Timestamped output with speaker labels
- 📝 Clean, formatted transcript output

## Prerequisites

- Python 3.12+
- [FFmpeg](https://ffmpeg.org/download.html) installed on your system
- CUDA-compatible GPU (recommended) or CPU for processing

## Installation

1. Clone the repository
2. Install Poetry (if not already installed)

3. Install dependencies:
```bash
poetry install
```
4. Install the package
```bash
pip install dist/transcriber-0.1.0-py3-none-any.whl

or 

pip install .

```

## Usage

### Basic Usage

```python
from transcriber import AudioTranscriber
transcriber = AudioTranscriber(
whisper_model_name="small.en", 
device="cuda" # Use "cpu" if no GPU is available
)

transcriber.transcribe(
audio_file="path/to/your/audio.mp3",
output_file="path/to/save/transcription.txt"
)
```

### Output Format

The transcription output is formatted as follows and saved to the output .txtfile:

```
[ HH:MM:SS -- HH:MM:SS ] SPEAKER_1 : Transcribed text for speaker 1
[ HH:MM:SS -- HH:MM:SS ] SPEAKER_2 : Transcribed text for speaker 2
...
```

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Transcription-diarization-whisper-pyannote](https://github.com/Mastering-Python-GT/Transcription-diarization-whisper-pyannote)
