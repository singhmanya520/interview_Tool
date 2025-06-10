
import streamlit as st
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from faster_whisper import WhisperModel
from pydub import AudioSegment, silence
from transformers import pipeline
import pandas as pd

st.title("🎤 Mock Interview Analyzer")
uploaded_file = st.file_uploader("Upload your interview audio/video file", type=["mp4", "mp3", "wav"])

def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.replace(".mp4", ".wav").replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_audio(audio_path):
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, beam_size=5)
    transcription = " ".join([seg.text for seg in segments])
    return transcription, list(segments)

def detect_emotion(audio_path):
    try:
        classifier = pipeline(
            "audio-classification",
            model="superb/wav2vec2-base-superb-er",
            device=-1
        )
        result = classifier(audio_path)
        emotion = result[0]['label']
        score = result[0]['score']
        return emotion, round(score, 2)
    except Exception as e:
        return "Unknown", 0.0

def count_filler_words(transcript):
    filler_words = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically']
    words = transcript.lower().split()
    return {fw: words.count(fw) for fw in filler_words if words.count(fw) > 0}

def compute_speech_rate(transcript, duration_seconds):
    word_count = len(transcript.split())
    words_per_minute = (word_count / duration_seconds) * 60
    return round(words_per_minute, 2)

def personal_pronoun_ratio(transcript):
    words = transcript.lower().split()
    return {
        "I": words.count("i"),
        "we": words.count("we"),
        "you": words.count("you")
    }

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.audio(file_path)

    audio_path = convert_to_wav(file_path)

    try:
        duration = librosa.get_duration(path=audio_path)
    except Exception:
        st.error("Could not read audio duration. Please check your file.")
        st.stop()

    if duration < 1:
        st.error("Audio is too short or corrupted.")
        st.stop()

    try:
        transcript, segments = transcribe_audio(audio_path)
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        st.stop()

    if not segments:
        st.error("No segments returned by transcription model.")
        st.stop()

    st.subheader("📝 Transcript")
    st.write(transcript)

    emotion, tone_score = detect_emotion(audio_path)
    st.subheader("😃 Detected Emotion")
    st.write(f"**{emotion}** (Confidence: {tone_score})")

    filler_counts = count_filler_words(transcript)
    if filler_counts:
        st.subheader("⛔ Filler Words Detected")
        st.write(filler_counts)
    else:
        st.subheader("✅ No common filler words detected!")

    speech_rate = compute_speech_rate(transcript, duration)
    st.subheader("📊 Speech Rate")
    st.write(f"{speech_rate} words per minute")
    if speech_rate < 110:
        st.info("Your speech rate is a bit slow. Slow speech can sometimes come across as hesitant.")
    elif speech_rate > 170:
        st.warning("Your speech rate is quite fast. Try slowing down to ensure clarity.")
    else:
        st.success("Great pace! Your speech rate is within the ideal range for interviews.")

    st.subheader("🔍 Personal Pronoun Usage")
    pronouns = personal_pronoun_ratio(transcript)
    st.write(pronouns)
    st.caption("Balance between 'I', 'we', and 'you' helps convey confidence and teamwork.")

    st.subheader("📊 Speech Rate Over Time")
    try:
        times = [seg.start for seg in segments if seg.end - seg.start > 0]
        lengths = [seg.end - seg.start for seg in segments if seg.end - seg.start > 0]
        words = [len(seg.text.split()) for seg in segments if seg.end - seg.start > 0]
        words_per_second = [w / l for w, l in zip(words, lengths)]

        if not times or not words_per_second:
            st.warning("Not enough valid segment data to generate graph.")
        else:
            fig, ax = plt.subplots()
            ax.plot(times, words_per_second, label="Speech Rate")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Words per second")
            ax.set_title("Speech Rate Over Time")
            st.pyplot(fig)
            st.caption("This chart shows how your speaking pace varied throughout the interview.")
    except Exception as e:
        st.error(f"Error generating graph: {str(e)}")
