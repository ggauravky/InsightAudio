from utils.audioProcessor import process_input
from core.transcriber import transcribe_all

source="https://youtu.be/YtnigjSD2BU?si=sxAIJgIq_9jBUEbk"

chunks=process_input(source)
transcription=transcribe_all(chunks)
print("Transcription Result:")
print(transcription)
