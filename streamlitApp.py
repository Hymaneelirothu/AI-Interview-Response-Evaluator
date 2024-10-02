import streamlit as st
import whisper
from moviepy.editor import VideoFileClip
from tempfile import NamedTemporaryFile
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import os

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

# Define criteria for scoring responses
criteria = {
    "technical": ["machine learning", "data", "preprocess", "decision tree", "SVM", "neural network", "hyperparameter"],
    "problem_solving": ["cross-validation", "grid search", "evaluate", "optimize", "performance"],
    "communication": ["I would", "then", "and", "also"]
}

# Function to encode a response using BERT
def encode_response(response):
    inputs = tokenizer(response, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

# Function to score the response based on predefined criteria
def score_response(response, criteria):
    scores = {}
    for criterion, keywords in criteria.items():
        scores[criterion] = sum([1 for word in keywords if word in response.lower()]) / len(keywords)
    return scores

# Function to rank candidates by average score
def rank_candidates(candidates):
    for candidate in candidates:
        avg_score = np.mean(list(candidate['scores'].values()))
        candidate['avg_score'] = avg_score
    ranked_candidates = sorted(candidates, key=lambda x: x['avg_score'], reverse=True)
    return ranked_candidates

# Function to extract audio from the video and perform transcription using Whisper
def transcribe_video(video_file):
    # Save the uploaded file to a temporary location
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(video_file.read())  # Write video file to the temp file
        temp_video_path = temp_video_file.name

    # Load the video and extract audio
    video = VideoFileClip(temp_video_path)
    audio_file = "audio.wav"
    video.audio.write_audiofile(audio_file)
    
    # Perform transcription with Whisper
    whisper_model = whisper.load_model("base")  # Ensure model is loaded here
    transcription = whisper_model.transcribe(audio_file)

    # Clean up temporary files
    os.remove(audio_file)
    os.remove(temp_video_path)

    return transcription['text']

# Streamlit app
st.title("AI Role Candidate Screening via Video Interview")

# Input for the number of candidates
num_candidates = st.number_input("Enter the number of candidates:", min_value=1, max_value=10, value=1)

mock_interviews = []
for i in range(num_candidates):
    video_file = st.file_uploader(f"Upload interview video for Candidate {i+1}:", type=["mp4", "mov", "avi"], key=f"video_{i}")
    if video_file:
        st.write(f"Processing video for Candidate {i+1}...")
        transcription = transcribe_video(video_file)
        st.write(f"Transcript for Candidate {i+1}: {transcription}")
        mock_interviews.append({"name": f"Candidate {i+1}", "response": transcription})

# Analyze the candidates when the user clicks the "Analyze" button
if st.button('Analyze Responses'):
    if mock_interviews:
        # Encode and score each candidate
        scored_candidates = []
        for candidate in mock_interviews:
            scores = score_response(candidate['response'], criteria)
            candidate['scores'] = scores
            candidate['encoded'] = encode_response(candidate['response'])
            scored_candidates.append(candidate)
        
        # Rank the candidates based on scores
        ranked_candidates = rank_candidates(scored_candidates)
        
        # Display the results
        st.write("### Candidate Rankings")
        for rank, candidate in enumerate(ranked_candidates, 1):
            st.write(f"**Rank {rank}: {candidate['name']}**")
            st.write(f"Average Score: {candidate['avg_score']:.2f}")
            st.write(f"Scores: {candidate['scores']}")
    else:
        st.write("Please upload videos for all candidates.")
