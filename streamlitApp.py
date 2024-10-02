import streamlit as st
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import speech_recognition as sr
import tempfile
import os

# Mock function to simulate scoring based on predefined criteria
def evaluate_interview(transcript):
    # For simplicity, let's score based on the length of the response.
    # You can replace this logic with your own AI-based evaluation logic.
    word_count = len(transcript.split())
    if word_count > 50:
        return 10  # Excellent
    elif word_count > 30:
        return 7   # Good
    elif word_count > 15:
        return 5   # Average
    else:
        return 2   # Needs Improvement

# Function to transcribe audio using SpeechRecognition
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    
    # Load the audio file using SpeechRecognition
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            # Use Google Web Speech API for transcription
            transcription = recognizer.recognize_google(audio_data)
            return transcription
        except sr.UnknownValueError:
            return "Audio unintelligible"
        except sr.RequestError:
            return "Could not request results from the speech recognition service"

# Function to extract audio from the video and perform transcription
def transcribe_video(video_file):
    # Save the uploaded video file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(video_file.read())
        tmp_filename = tmp_file.name
    
    # Process the video using the saved temporary file
    video = VideoFileClip(tmp_filename)
    
    # Create a temporary audio file
    audio_file = "audio.wav"
    video.audio.write_audiofile(audio_file)

    # Now, pass `audio_file` to your transcription function
    transcription = transcribe_audio(audio_file)

    # Clean up the temporary files
    os.remove(tmp_filename)
    os.remove(audio_file)

    return transcription

# Streamlit App Interface
st.title("AI Interview Response Evaluator")

# Collecting interview videos and evaluating responses
mock_interviews = []
num_candidates = st.number_input("Enter number of candidates", min_value=1, max_value=10, value=1)

for i in range(num_candidates):
    st.write(f"Candidate {i+1}")
    video_file = st.file_uploader(f"Upload interview video for Candidate {i+1}", type=["mp4", "mov", "avi"])
    
    if video_file:
        st.write(f"Processing video for Candidate {i+1}...")
        transcription = transcribe_video(video_file)
        st.write(f"Transcript for Candidate {i+1}: {transcription}")
        
        # Evaluate the transcript and give a score
        score = evaluate_interview(transcription)
        st.write(f"Score for Candidate {i+1}: {score}/10")
        
        # Append mock interview results
        mock_interviews.append({"name": f"Candidate {i+1}", "response": transcription, "score": score})

# Show final results
if len(mock_interviews) > 0:
    st.write("Final Results:")
    for interview in mock_interviews:
        st.write(f"{interview['name']}: {interview['score']}/10")
