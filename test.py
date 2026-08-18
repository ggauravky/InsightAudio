from utils.audioProcessor import process_input
from core.transcriber import transcribe_all

source="https://youtu.be/UabBYexBD4k?si=znlirK2qYnmmlmmX"

chunks=process_input(source)
transcription=transcribe_all(chunks)
print("Transcription Result:")
print(transcription)
