import streamlit as st
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import subprocess
from pydub import AudioSegment, silence
import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

CHUNK_DURATION = 10
FILLER_WORDS = ["um", "uh", "like", "you know", "so", "actually", "basically", "right"]

st.title("🎤 Mock Interview Analyzer")

uploaded_file = st.file_uploader("Upload your interview video (.mp4)", type=["mp4"])

if uploaded_file is not None:
    with st.spinner("Processing video..."):
        video_path = "uploaded_video.mp4"
        audio_path = "interview_audio.wav"

        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        subprocess.call(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', audio_path])

        # Whisper Transcription
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True)
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

        st.subheader("📈 Speech Rate Over Time")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(range(0, CHUNK_DURATION * bins, CHUNK_DURATION), speech_rate, marker='o')
        ax1.set_title("Words Spoken Every 10 Seconds")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Words")
        ax1.grid(True)
        st.pyplot(fig1)

        avg_rate = round(np.mean(speech_rate), 1)
        if avg_rate < 15:
            pacing_msg = "🟡 You're speaking a bit slowly. Consider picking up the pace slightly."
        elif avg_rate <= 30:
            pacing_msg = "🟢 Your pacing is clear and well-controlled."
        else:
            pacing_msg = "🔴 You're speaking quite quickly. Try pausing occasionally."

        st.markdown(f"**Average Speech Rate:** {avg_rate} words/10s  \n{pacing_msg}")


        # Silence detection
        audio = AudioSegment.from_wav(audio_path)
        silent_chunks = silence.detect_silence(audio, min_silence_len=1000, silence_thresh=audio.dBFS - 14)
        silent_seconds = [(start / 1000.0, stop / 1000.0) for start, stop in silent_chunks]

        # Filler and pronoun counts
        filler_count = sum(len(re.findall(rf"\\b{re.escape(word)}\\b", full_text)) for word in FILLER_WORDS)
        pronoun_counts = {
            "i": len(re.findall(r"\\bi\\b", full_text)),
            "we": len(re.findall(r"\\bwe\\b", full_text)),
            "you": len(re.findall(r"\\byou\\b", full_text)),
        }

        st.subheader("🔊 Confidence and Tone Analysis")
        emotion_model = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        y, sr = librosa.load(audio_path, sr=None)
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

        fig2, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(times, confidence_vals, label="Confidence Level", marker='o')
        ax1.set_ylabel("Confidence")
        ax1.set_xlabel("Time")
        ax1.tick_params(axis='x', rotation=45)

        ax2 = ax1.twinx()
        ax2.bar(times, variation, alpha=0.3, color='orange', label="Tone Variation")
        ax2.set_ylabel("Tone Variation")

        for i, m in enumerate(tone_metrics):
            if m['monotone']:
                ax1.axvline(x=times[i], color='red', linestyle='--', alpha=0.5)
                ax1.text(times[i], 0, 'Monotone', rotation=90, color='red', fontsize=8, ha='center')

        fig2.legend(loc="upper right")
        ax1.set_title("Interview Confidence and Tone Analysis")
        st.pyplot(fig2)

        avg_conf = round(np.mean(confidence_vals), 2)
        monotone_count = sum(m['monotone'] for m in tone_metrics)

        st.markdown(f"""**💬 Average Confidence:** {avg_conf}  
**Monotone Segments:** {monotone_count}""")

        st.subheader("📋 Interview Summary Table")

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
                "0–3 ideal (≤8 okay)", "5–12", "3–8", "1–4", "0–2 ideal (≤5 okay)"
            ],
            "Feedback": [
                filler_fb, pronoun_fb, "", "", pause_fb
            ],
            "Status": [
                filler_status, pronoun_status, "", "", pause_status
            ]
        })

        st.dataframe(feedback_df, use_container_width=True)

        st.subheader("🧠 LLM Feedback")
        tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
        model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M")
        llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

        tone_summary = ", ".join(f"{m['start']}s: {m['top_emotion']}" for m in tone_metrics)
        prompt = f'''
You are a warm and constructive interview coach. Based on this tone data:

Tone Timeline: {tone_summary}
Average Confidence: {avg_conf}
Monotone Segments: {monotone_count}

1. Write one positive observation about the candidate’s vocal tone.
2. Write one constructive suggestion.
3. Suggest two specific tips that could help them improve.

Each point should appear on a new line.
'''
        llm_feedback = llm(prompt, max_new_tokens=300, do_sample=False)[0]['generated_text']
        st.markdown(f"""**LLM Feedback:**  
{llm_feedback}""")
