
import re
import streamlit as st
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import whisper
from faster_whisper import WhisperModel
from pydub import AudioSegment, silence
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import pandas as pd


CHUNK_DURATION = 10  # seconds for chunking audio
FILLER_WORDS = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically']


st.title("🎤 Mock Interview Analyzer")
uploaded_file = st.file_uploader("Upload your interview audio/video file", type=["mp4", "mp3", "wav"])

def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.replace(".mp4", ".wav").replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_audio(audio_path):
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    transcription = " ".join([seg.text for seg in segments])
    return transcription, list(segments)

def transcribe_audio_legacy(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)
    full_text = result['text'].lower()
    segments = result['segments']
    return full_text, segments

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

    print("audio_path:", audio_path)
    if not os.path.exists(audio_path):
        st.error("Converted audio file not found. Please check your upload.")
        st.stop()

    try:
        duration = librosa.get_duration(path=audio_path)
    except Exception:
        st.error("Could not read audio duration. Please check your file.")
        st.stop()

    if duration < 1:
        st.error("Audio is too short or corrupted.")
        st.stop()

    try:
        transcript, segments = transcribe_audio_legacy(audio_path)
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

    """
    Plotting code 
    """
    # try:
    #     times = [seg.start for seg in segments if seg.end - seg.start > 0]
    #     lengths = [seg.end - seg.start for seg in segments if seg.end - seg.start > 0]
    #     words = [len(seg.text.split()) for seg in segments if seg.end - seg.start > 0]
    #     words_per_second = [w / l for w, l in zip(words, lengths)]

    #     if not times or not words_per_second:
    #         st.warning("Not enough valid segment data to generate graph.")
    #     else:
    #         fig, ax = plt.subplots()
    #         ax.plot(times, words_per_second, label="Speech Rate")
    #         ax.set_xlabel("Time (s)")
    #         ax.set_ylabel("Words per second")
    #         ax.set_title("Speech Rate Over Time")
    #         st.pyplot(fig)
    #         st.caption("This chart shows how your speaking pace varied throughout the interview.")
    # except Exception as e:
    #     st.error(f"Error generating graph: {str(e)}")

    try:
        word_timings = []
        for seg in segments:
            for word_data in seg.get("words", []):
                word_timings.append({
                    "start": word_data["start"],
                    "end": word_data["end"],
                    "word": word_data["word"].lower()
                })
        
        duration = word_timings[-1]["end"] if word_timings else 0
        bins = int(duration // 10) + 1
        speech_rate = [0] * bins 
        for word in word_timings:
            bin_index = int(word["start"] // 10)
            # if bin_index < bins:
            speech_rate[bin_index] += 1

        plt.figure(figsize=(10, 4))
        plt.plot(range(0, 10 * bins, 10), speech_rate, marker='o', linestyle='-')
        plt.title("Speech Rate Over Time (Words per 10s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Words Spoken")
        plt.grid(True)
        plt.tight_layout()
        fig = plt.gcf()
        st.pyplot(fig)
        st.caption("This chart shows how your speaking pace varied throughout the interview.")
    except Exception as e:
        st.error(f"Error generating graph: {str(e)}")


    # Dynamic Feedback on Speech Rate
    avg_rate = round(np.mean(speech_rate), 1)
    st.write("\n💬 INTERPRETATION: SPEECH PACING")
    st.write(f"Average speech rate: {avg_rate} words per 10s.")
    if avg_rate < 15:
        st.write("🟡 You're speaking a bit slowly, which might come across as hesitant or unsure to interviewers. Consider picking up the pace slightly to maintain engagement.")
    elif avg_rate <= 30:
        st.write("🟢 Your pacing is clear and well-controlled — it gives a sense of calm confidence, which is great for interviews.")
    else:
        st.write("🔴 You're speaking quite quickly. While energy is good, interviewers may miss key points if you rush. Try pausing occasionally to let ideas land.")

    # STEP 7: Pause Detection
    audio = AudioSegment.from_wav(audio_path)
    silent_chunks = silence.detect_silence(audio, min_silence_len=1000, silence_thresh=audio.dBFS - 14)
    silent_seconds = [(start / 1000.0, stop / 1000.0) for start, stop in silent_chunks]

    # STEP 8: Filler Words + Pronouns
    filler_count = sum(len(re.findall(rf"\b{re.escape(word)}\b", transcript)) for word in FILLER_WORDS)
    pronoun_counts = {
        "i": len(re.findall(r"\bi\b", transcript)),
        "we": len(re.findall(r"\bwe\b", transcript)),
        "you": len(re.findall(r"\byou\b", transcript)),
    }

    # STEP 9: Confidence + Tone
    emotion_model = pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )
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

    # STEP 10: Plot Confidence + Tone Variation
    times = [f"{m['start']}s" for m in tone_metrics]
    confidence = [m['confidence'] for m in tone_metrics]
    variation = [m['variation'] for m in tone_metrics]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(times, confidence, label="Confidence Level", marker='o')
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

    fig.legend(loc="upper right")
    plt.title("Interview Confidence and Tone Analysis")
    plt.tight_layout()
    # plt.show()
    fig = plt.gcf()
    st.subheader("📊 Confidence and Tone Analysis")
    st.pyplot(fig)
    st.caption("This chart shows your confidence level and tone variation over time. Monotone segments are highlighted.")

    # STEP 11: Interpretation for Confidence + Tone
    avg_conf = round(np.mean(confidence), 2)
    monotone_count = sum(m['monotone'] for m in tone_metrics)
    st.write("\n💬 INTERPRETATION: VOCAL TONE & CONFIDENCE")
    st.write(f"Average confidence: {avg_conf}")
    if avg_conf >= 0.8:
        st.write("🟢 Your tone sounds confident and expressive — this creates a strong presence and makes your message more compelling.")
    elif avg_conf >= 0.6:
        st.write("🟡 Your tone feels fairly confident, which is a good start. With just a bit more vocal energy or emphasis on key points, you'll sound even more engaging.")
    else:
        st.write("🔴 Your tone came across as flat or uncertain at times. Try to vary your pitch and energy — this helps you sound more assured and keep the listener's attention.")

    if monotone_count == 0:
        st.write("🟢 Great vocal variety throughout — your tone was dynamic and expressive.")
    elif monotone_count <= 2:
        st.write("🟡 A couple of segments sounded monotone. Try emphasizing important words or using pitch variation to maintain interest.")
    else:
        st.write("🔴 There were several monotone segments. Practicing with emphasis and emotion can really help make your delivery more compelling.")

    # STEP 12: Summary Table
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

    st.write("\n📋 FINAL INTERVIEW COACH SUMMARY TABLE")
    st.dataframe(feedback_df)

    # STEP 13: LLM Feedback (Natural, Split by Lines)
    tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
    model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M")
    llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device="cpu")

    tone_summary = ", ".join(f"{m['start']}s: {m['top_emotion']}" for m in tone_metrics)
    prompt = f"""
    You are a warm and constructive interview coach. Based on this tone data:

    Tone Timeline: {tone_summary}
    Average Confidence: {avg_conf}
    Monotone Segments: {monotone_count}

    1. Write one positive observation about the candidate’s vocal tone.
    2. Write one constructive suggestion.
    3. Suggest two specific tips that could help them improve.

    Each point should appear on a new line.
    """

    llm_feedback = llm(prompt, max_new_tokens=300, do_sample=False)[0]['generated_text']

    print("llm_feedback", llm_feedback)

    st.write("\n🧠 LLM FEEDBACK — VOCAL TONE COACHING")
    # for line in llm_feedback.strip().split("1."):
    #     if line.strip():
    #         st.write("1." + line.strip().replace("2.", "\n2.").replace("3.", "\n3.").replace("4.", "\n4."))

    # st.write(llm_feedback.strip())
    llm_feedback = llm_feedback.strip().replace("1.", "\n1.").replace("2.", "\n2.").replace("3.", "\n3.")

    for line in llm_feedback.split("\n"):
        st.write(line.strip())
