# app_streamlit.py

import streamlit as st
import whisper
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from pydub import AudioSegment, silence
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import librosa
import soundfile as sf
import pandas as pd

# App title
st.title("🎤 Mock Interview Feedback Tool")

# Upload video
uploaded_file = st.file_uploader("Upload your mock interview video (.mp4)", type=["mp4"])

if uploaded_file:
    with open("interview.mp4", "wb") as f:
        f.write(uploaded_file.read())

    VIDEO_PATH = "interview.mp4"
    AUDIO_PATH = "interview_audio.wav"
    CHUNK_DURATION = 10
    FILLER_WORDS = ["um", "uh", "like", "you know", "so", "actually", "basically", "right"]

    st.info("🔄 Extracting audio...")
    video = VideoFileClip(VIDEO_PATH)
    video.audio.write_audiofile(AUDIO_PATH, codec='pcm_s16le')

    model = whisper.load_model("base")
    result = model.transcribe(AUDIO_PATH, word_timestamps=True)
    full_text = result["text"].lower()
    segments = result["segments"]

    word_timings = []
    for segment in segments:
        for word_data in segment.get("words", []):
            word_timings.append({
                "word": word_data["word"].lower(),
                "start": word_data["start"],
                "end": word_data["end"]
            })

    duration = word_timings[-1]["end"] if word_timings else 0
    bins = int(duration // CHUNK_DURATION) + 1
    speech_rate = [0] * bins
    for word in word_timings:
        bin_index = int(word["start"] // CHUNK_DURATION)
        speech_rate[bin_index] += 1

    # Speech rate plot
    st.subheader("🗣️ Speech Rate Over Time")
    fig1, ax1 = plt.subplots()
    ax1.plot(range(0, CHUNK_DURATION * bins, CHUNK_DURATION), speech_rate, marker='o')
    ax1.set_title("Speech Rate Over Time (Words per 10s)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Words Spoken")
    ax1.grid(True)
    st.pyplot(fig1)

    avg_rate = round(np.mean(speech_rate), 1)
    st.markdown("### 💬 Interpretation: Speech Pacing")
    if avg_rate < 15:
        st.write(f"🟡 You're speaking slowly ({avg_rate} w/10s). This can sometimes be perceived as uncertainty or hesitation.")
    elif avg_rate <= 30:
        st.write(f"🟢 Your pace ({avg_rate} w/10s) is in the ideal range. This reflects clarity and confidence.")
    else:
        st.write(f"🔴 You're speaking quickly ({avg_rate} w/10s). Consider slowing down to emphasize key points.")

    audio = AudioSegment.from_wav(AUDIO_PATH)
    silent_chunks = silence.detect_silence(audio, min_silence_len=1000, silence_thresh=audio.dBFS - 14)
    silent_seconds = [(start / 1000.0, stop / 1000.0) for start, stop in silent_chunks]

    filler_count = sum(len(re.findall(rf"\b{re.escape(word)}\b", full_text)) for word in FILLER_WORDS)
    pronoun_counts = {
        "i": len(re.findall(r"\bi\b", full_text)),
        "we": len(re.findall(r"\bwe\b", full_text)),
        "you": len(re.findall(r"\byou\b", full_text)),
    }

    st.info("🔍 Analyzing tone and emotion...")
    emotion_model = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    y, sr = librosa.load(AUDIO_PATH, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)

    tone_metrics = []
    for i in range(0, int(total_duration), CHUNK_DURATION):
        start = i
        end = min(i + CHUNK_DURATION, int(total_duration))
        chunk_y = y[int(start * sr):int(end * sr)]
        if len(chunk_y) < sr: continue
        sf.write("chunk.wav", chunk_y, sr)
        result = emotion_model("chunk.wav")
        top_emotions = sorted(result, key=lambda x: x['score'], reverse=True)
        confidence = top_emotions[0]['score']
        variation = np.std(chunk_y)
        is_monotone = variation < 0.01
        tone_metrics.append({
            "start": start,
            "confidence": round(confidence, 3),
            "variation": round(variation, 5),
            "monotone": is_monotone,
            "top_emotion": top_emotions[0]['label']
        })

    times = [f"{m['start']}s" for m in tone_metrics]
    confidence_vals = [m['confidence'] for m in tone_metrics]
    variation = [m['variation'] for m in tone_metrics]

    st.subheader("🎙️ Tone Confidence and Variation")
    fig2, ax2 = plt.subplots()
    ax2.plot(times, confidence_vals, label="Confidence", marker='o')
    ax2.set_ylabel("Confidence")
    ax2.set_xlabel("Time")
    ax2.tick_params(axis='x', rotation=45)

    ax3 = ax2.twinx()
    ax3.bar(times, variation, alpha=0.3, color='orange', label="Tone Variation")
    ax3.set_ylabel("Tone Variation")

    for i, m in enumerate(tone_metrics):
        if m['monotone']:
            ax2.axvline(x=times[i], color='red', linestyle='--', alpha=0.5)
            ax2.text(times[i], 0, 'Monotone', rotation=90, color='red', fontsize=8, ha='center')

    fig2.legend(loc="upper right")
    st.pyplot(fig2)

    avg_conf = round(np.mean(confidence_vals), 2)
    monotone_count = sum(m['monotone'] for m in tone_metrics)
    st.markdown("### 💬 Interpretation: Vocal Tone & Confidence")
    if avg_conf >= 0.8:
        st.write("🟢 Your tone is strong and expressive. Interviewers will likely perceive you as confident and engaging.")
    elif avg_conf >= 0.6:
        st.write("🟡 Your tone is fairly confident. With a bit more energy and emphasis, it can sound even more compelling.")
    else:
        st.write("🔴 Your tone sounds flat or hesitant. Practice vocal projection and expressiveness to improve clarity.")

    if monotone_count == 0:
        st.write("🟢 Excellent tone variation — no segments sounded monotone.")
    elif monotone_count <= 2:
        st.write("🟡 Some segments sounded monotone — consider practicing with more pitch and emotion.")
    else:
        st.write("🔴 Multiple monotone segments detected — try to vary your pitch and tone to stay engaging.")

    st.subheader("📋 Final Interview Summary")

    def interpret_filler(count):
        if count <= 3: return "Great! Minimal filler use.", "✅"
        elif count <= 8: return "Moderate use — try reducing further.", "⚠️"
        else: return "High — practice smoother transitions.", "❌"

    def interpret_pronouns(i, we):
        if i > 10 and we < 3: return "Try balancing 'I' with more 'we'.", "❌"
        elif we > i: return "Strong collaboration emphasis.", "✅"
        else: return "Balanced — well done.", "⚠️"

    def interpret_pauses(p):
        if p <= 2: return "Good pacing and minimal long pauses.", "✅"
        elif p <= 5: return "Try to reduce long pauses.", "⚠️"
        else: return "Too many pauses — rehearse for flow.", "❌"

    filler_fb, filler_status = interpret_filler(filler_count)
    pronoun_fb, pronoun_status = interpret_pronouns(pronoun_counts["i"], pronoun_counts["we"])
    pause_fb, pause_status = interpret_pauses(len(silent_seconds))

    ideal_ranges = {
        "Filler Words Used": "0–3 ideal (≤8 okay)",
        "Times You Said 'I'": "5–12",
        "Times You Said 'We'": "3–8",
        "Times You Said 'You'": "1–4",
        "Detected Pauses >1s": "0–2 ideal (≤5 okay)"
    }

    feedback_df = pd.DataFrame({
        "Metric": [
            "Filler Words Used", "Times You Said 'I'", "Times You Said 'We'",
            "Times You Said 'You'", "Detected Pauses >1s"
        ],
        "Your Count": [
            filler_count, pronoun_counts["i"], pronoun_counts["we"],
            pronoun_counts["you"], len(silent_seconds)
        ],
        "Ideal Range": [
            ideal_ranges["Filler Words Used"],
            ideal_ranges["Times You Said 'I'"],
            ideal_ranges["Times You Said 'We'"],
            ideal_ranges["Times You Said 'You'"],
            ideal_ranges["Detected Pauses >1s"]
        ],
        "Feedback": [
            filler_fb, pronoun_fb, "", "", pause_fb
        ],
        "Status": [
            filler_status, pronoun_status, "", "", pause_status
        ]
    })

    st.dataframe(feedback_df)
