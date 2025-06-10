import streamlit as st
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from faster_whisper import WhisperModel
from pydub import AudioSegment, silence
from transformers import pipeline
import librosa
import soundfile as sf
import re

# --- STEP 1: File Upload ---
st.title("🎤 Mock Interview Analyzer")
uploaded_file = st.file_uploader("Upload your interview video (.mp4 or .mov)", type=["mp4", "mov"])

# --- STEP 2: Audio Extraction using pydub ---
def extract_audio_from_video(video_path):
    audio_path = "interview_audio.wav"
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_path, format="wav")
    return audio_path

# --- STEP 3: Speech-to-Text ---
def transcribe_audio_faster_whisper(audio_path):
    model_size = "base"
    model = WhisperModel(model_size, compute_type="int8", cpu_threads=4)
    segments, _ = model.transcribe(audio_path)
    full_text = ""
    timestamps = []
    for segment in segments:
        full_text += segment.text + " "
        timestamps.append((segment.start, segment.end, segment.text))
    return full_text.strip(), timestamps

# --- STEP 4: Filler Word Detection ---
def count_filler_words(text):
    filler_words = ["um", "uh", "like", "you know", "so", "actually"]
    text_lower = text.lower()
    count = sum(text_lower.count(filler) for filler in filler_words)
    return count

# --- STEP 5: Personal Pronoun Ratio ---
def count_pronouns(text):
    text = text.lower()
    return text.count("i"), text.count("we"), text.count("you")

# --- STEP 6: Pause Detection from Audio ---
def detect_pauses(audio_path):
    sound = AudioSegment.from_wav(audio_path)
    silent_chunks = silence.detect_silence(sound, min_silence_len=1000, silence_thresh=sound.dBFS-14)
    return len(silent_chunks)

# --- STEP 7: Speech Pacing ---
def calculate_speech_rate(transcript, timestamps):
    total_words = len(transcript.split())
    total_time = timestamps[-1][1] - timestamps[0][0]
    rate = total_words / (total_time / 10)
    return round(rate, 1)

# --- STEP 8: Emotion Detection ---
def detect_emotion(audio_path):
    classifier = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    data, rate = librosa.load(audio_path, sr=16000)
    sf.write("temp_emotion.wav", data, rate)
    results = classifier("temp_emotion.wav")
    top_result = max(results, key=lambda x: x['score'])
    return top_result['label'], round(top_result['score'], 2)

# --- STEP 9: Tone Interpretation ---
def interpret_tone(confidence_score):
    if confidence_score >= 0.5:
        return "🟢 Your tone is fairly confident and expressive — interviewers will likely respond well to this."
    elif 0.2 <= confidence_score < 0.5:
        return "🟡 Your tone sounds neutral. Consider adding vocal variation to maintain engagement."
    else:
        return "🔴 Your tone sounds flat or unsure — try projecting more."

# --- STEP 10: Speech Rate Feedback ---
def interpret_speech_rate(rate):
    if rate >= 25 and rate <= 35:
        return "🟢 You're in the ideal range. Good clarity and confidence."
    elif rate < 25:
        return "🟡 Your current rate is under the average. Speaking slowly can sometimes come off as hesitancy or uncertainty."
    else:
        return "🔴 You're speaking quite fast — this may make you sound nervous or unclear."

# --- STEP 11: LLM-style Dynamic Feedback ---
def generate_llm_feedback(pronouns, filler_count, pause_count, tone_score, speech_rate):
    feedback = []

    # Pronouns
    if pronouns[0] > 15 and pronouns[1] < 3:
        feedback.append("You used 'I' quite a lot. Try to incorporate more team-oriented language like 'we'.")

    # Filler words
    if filler_count > 8:
        feedback.append(f"You used filler words {filler_count} times — this can reduce your clarity. Practice slowing down and pausing instead of using fillers.")
    elif 4 <= filler_count <= 8:
        feedback.append(f"You used filler words {filler_count} times. It’s okay, but keep an eye on it.")

    # Pauses
    if pause_count > 5:
        feedback.append("There were many long pauses — rehearsing your responses could improve your flow.")

    # Tone
    if tone_score < 0.2:
        feedback.append("Your tone was flat at times. Projecting your voice more could make you sound more confident.")

    # Speech rate
    if speech_rate < 25:
        feedback.append("You spoke a bit slowly — try to maintain a steady pace to convey energy.")
    elif speech_rate > 35:
        feedback.append("You spoke quickly — slowing down slightly could help with clarity.")

    return feedback

# --- MAIN ---
if uploaded_file is not None:
    with st.spinner("⏳ Processing video..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        audio_path = extract_audio_from_video(tmp_path)
        transcript, timestamps = transcribe_audio_faster_whisper(audio_path)
        filler_count = count_filler_words(transcript)
        pronouns = count_pronouns(transcript)
        pause_count = detect_pauses(audio_path)
        speech_rate = calculate_speech_rate(transcript, timestamps)
        emotion, tone_score = detect_emotion(audio_path)

    st.header("📝 Transcript")
    st.write(transcript)

    st.header("📊 Interview Feedback Summary")
    st.markdown(f"""
    - **Filler Words Used:** {filler_count}
    - **Pronouns Used:** I = {pronouns[0]}, We = {pronouns[1]}, You = {pronouns[2]}
    - **Detected Pauses (>1s):** {pause_count}
    - **Speech Rate:** {speech_rate} words per 10 seconds
    - **Vocal Emotion:** {emotion} (confidence score: {tone_score})
    """)

    st.header("💬 Interpretation")
    st.markdown(interpret_speech_rate(speech_rate))
    st.markdown(interpret_tone(tone_score))

    st.header("🧠 LLM-Style Feedback")
    feedback_list = generate_llm_feedback(pronouns, filler_count, pause_count, tone_score, speech_rate)
    for point in feedback_list:
        st.markdown(f"- {point}")
