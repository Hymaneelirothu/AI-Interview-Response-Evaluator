import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define criteria for scoring
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
        scores[criterion] = sum([1 for word in keywords if word in response]) / len(keywords)
    return scores

# Function to rank candidates by average score
def rank_candidates(candidates):
    for candidate in candidates:
        avg_score = np.mean(list(candidate['scores'].values()))
        candidate['avg_score'] = avg_score
    ranked_candidates = sorted(candidates, key=lambda x: x['avg_score'], reverse=True)
    return ranked_candidates

# Streamlit app
st.title("AI Role Candidate Screening")

# Input for the number of candidates
num_candidates = st.number_input("Enter the number of candidates:", min_value=1, max_value=10, value=3)

# Create input fields for candidate names and responses
mock_interviews = []
for i in range(num_candidates):
    name = st.text_input(f"Enter the name of Candidate {i+1}:", key=f"name_{i}")
    response = st.text_area(f"Enter the interview response for {name}:", key=f"response_{i}")
    if name and response:
        mock_interviews.append({"name": name, "response": response})

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
        st.write("Please enter candidate responses.")
