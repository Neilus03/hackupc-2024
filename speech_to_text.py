import sys
import soundfile as sf
from transformers import pipeline
import torch
import warnings

#
def mp3_to_text(audio_file, target_language='en', output_file='transcript.txt'):
    # Check file extension and adjust loading method if needed
    file_extension = audio_file.split('.')[-1].lower()
    if file_extension not in ['mp3', 'wav']:
        raise ValueError("Unsupported file type. Please use MP3 or WAV files.")

    # Load the audio file using soundfile
    waveform, sample_rate = sf.read(audio_file, dtype='float32')
    
    # If the loaded audio is multi-channel (stereo), convert to mono by averaging the channels
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Initialize the Whisper model pipeline
    whisper = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    # Run the speech recognition with specified language handling
    result = whisper({
        "raw": waveform,
        "sampling_rate": sample_rate
    })  # This ensures transcription targets English

    # Extract the text
    text = result["text"]
    
    #save the text to the output file if specified
    if output_file:
        with open(output_file, "w") as f:
            f.write(text)
    return text

if __name__ == "__main__":
    audio_file = sys.argv[1]
    # Call the function to convert the audio file to text
    try:
        text = mp3_to_text(audio_file)
        with open("transcript.txt", "r") as f:
            print(f.read())
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
