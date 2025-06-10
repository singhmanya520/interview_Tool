import streamlit as st
from faster_whisper import WhisperModel
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from pydub import AudioSegment, silence
from transformers import pipeline
import librosa
import soundfile as sf
import pandas as pd
import tempfile

# UI setup
st.set_page_config(page_title="Mock Interview Analyzer", layout="wide")
st.title("🎤 Mock Interview Feedback Tool")
st.markdown("Upload a mock interview video (.mp4) to receive AI-powered feedback on pacing, tone, and delivery.")

# File upload
uploaded_file = st.file_uploader("Upload your interview video", type=["mp4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        VIDEO_PATH = tmp.name

    AUDIO_PATH = "interview_audio.wav"
    CHUNK_DURATION = 10  # in seconds
    FILLER_WORDS = ["um", "uh", "like", "you know", "so", "actually", "basically", "right"]

    st.info("🔄 Extracting audio...")
    video = VideoFileClip(VIDEO_PATH)
    video.audio.write_audiofile(AUDIO_PATH, codec='pcm_s16le')

    # Transcribe using faster-whisper
    st.info("🧠 Transcribing using faster-whisper...")
    model = WhisperModel("base", compute_type="int8")
    segments, info = model.transcribe(AUDIO_PATH, beam_size=5, word_timestamps=True)

    full_text = ""
    word_timings = []
    for segment in segments:
        full_text += segment.text.lower() + " "
        if segment.words:
            for word in segment.words:
                word_timings.append({
                    "word": word.word.lower(),
                    "start": word.start,
                    "end": word.end
                })

    duration = word_timings[-1]["end"] if word_timings else 0
    bins = int(duration // CHUNK_DURATION) + 1
    speech_rate = [0] * bins
    for word in word_timings:
        bin_index = int(word["start"] // CHUNK_DURATION)
        speech_rate[bin_index] += 1

    st.subheader("📊 Speech Rate Over Time")
    fig, ax = plt.subplots()
    ax.plot(range(len(speech_rate)), speech_rate, marker='o')
    ax.set_xlabel("Time Segment (10s intervals)")
    ax.set_ylabel("Words Spoken")
    ax.set_title("Speech Rate Across Interview")
    st.pyplot(fig)

    avg_rate = np.mean(speech_rate)
    st.markdown("#### 💬 Interpretation: Speech Pacing")
    st.write(f"**Average speech rate:** {avg_rate:.1f} words per 10 seconds.")
    if avg_rate < 20:
        st.warning("🟠 You spoke quite slowly. This can sometimes come across as uncertainty or hesitation.")
    elif avg_rate > 35:
        st.warning("🔴 You spoke quite fast. Try pausing more — fast pacing can affect clarity and confidence.")
    else:
        st.success("🟢 You're in the ideal range. Good clarity and confidence.")

    # Detect filler words
    st.subheader("📉 Filler Word Usage")
    filler_count = sum(full_text.count(f) for f in FILLER_WORDS)
    st.write(f"**Filler words used:** {filler_count}")
    if filler_count <= 3:
        st.success("✅ Great! Minimal filler use.")
    elif filler_count <= 8:
        st.warning("🟠 Acceptable, but try reducing filler words to sound more confident.")
    else:
        st.error("🔴 High filler usage — work on pausing naturally instead of using filler words.")

    # Analyze pauses
    st.subheader("⏸️ Detected Pauses")
    audio = AudioSegment.from_wav(AUDIO_PATH)
    silent_chunks = silence.detect_silence(audio, min_silence_len=1000, silence_thresh=audio.dBFS - 14)
    num_pauses = len(silent_chunks)
    st.write(f"**Pauses over 1 second:** {num_pauses}")
    if num_pauses <= 2:
        st.success("✅ Great flow and minimal hesitation.")
    elif num_pauses <= 5:
        st.warning("🟠 A few long pauses — consider rehearsing more for flow.")
    else:
        st.error("🔴 Too many pauses — this may signal uncertainty.")

    # Personal pronoun count
    st.subheader("🧠 Language Balance")
    i_count = full_text.count("i ")
    we_count = full_text.count("we ")
    you_count = full_text.count("you ")
    st.write(f"**'I' used:** {i_count}")
    st.write(f"**'We' used:** {we_count}")
    st.write(f"**'You' used:** {you_count}")
    if i_count > 15 and we_count < 3:
        st.warning("🟠 Try to balance 'I' with more 'we' to show teamwork.")

    # Tone and emotion estimation
    st.subheader("🎙️ Vocal Tone & Confidence")
    classifier = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    preds = classifier(AUDIO_PATH, top_k=5)
    labels = [p['label'] for p in preds]
    scores = [p['score'] for p in preds]
    avg_conf = np.mean(scores)
    st.write(f"**Top detected emotions:** {labels}")
    st.write(f"**Average confidence score:** {avg_conf:.2f}")
    if avg_conf < 0.2:
        st.error("🔴 Tone sounds flat or unsure — project more.")
    elif avg_conf < 0.5:
        st.warning("🟠 Moderate expressiveness — try emphasizing key points more.")
    else:
        st.success("🟢 Excellent variation — your tone shows confidence and engagement.")

    # Summary
    st.subheader("📋 Final Interview Coach Summary")
    summary_df = pd.DataFrame([
        {"Metric": "Filler Words Used", "Your Count": filler_count, "Ideal Range": "0–3 ideal (≤8 okay)",
         "Feedback": "Great! Minimal filler use." if filler_count <= 3 else
                     "Okay, but reduce fillers." if filler_count <= 8 else
                     "Too many filler words — practice smoother transitions.",
         "Status": "✅" if filler_count <= 3 else "⚠️" if filler_count <= 8 else "❌"},
        {"Metric": "'I' Count", "Your Count": i_count, "Ideal Range": "5–12",
         "Feedback": "Try balancing 'I' with more 'we' for teamwork.",
         "Status": "✅" if 5 <= i_count <= 12 else "❌"},
        {"Metric": "'We' Count", "Your Count": we_count, "Ideal Range": "3–8",
         "Feedback": "",
         "Status": "✅" if 3 <= we_count <= 8 else "⚠️"},
        {"Metric": "'You' Count", "Your Count": you_count, "Ideal Range": "1–4",
         "Feedback": "",
         "Status": "✅" if 1 <= you_count <= 4 else "⚠️"},
        {"Metric": "Pauses >1s", "Your Count": num_pauses, "Ideal Range": "0–2 ideal (≤5 okay)",
         "Feedback": "Too many pauses — rehearse for flow." if num_pauses > 5 else
                     "Some pauses, but acceptable." if num_pauses > 2 else
                     "Great flow!",
         "Status": "✅" if num_pauses <= 2 else "⚠️" if num_pauses <= 5 else "❌"},
    ])
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("---")
    st.markdown("Made with ❤️ for mock interview improvement.")
