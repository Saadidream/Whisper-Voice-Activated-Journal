import streamlit as st
import whisper
import tempfile
import os
import pandas as pd
import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load the Whisper model (using 'base' for speed; you can change to 'small' or 'medium' for higher accuracy)
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

# Initialize or retrieve the journal entries DataFrame from session_state
if "journal_df" not in st.session_state:
    st.session_state.journal_df = pd.DataFrame(columns=["Timestamp", "Transcription", "Sentiment"])

st.title("Voice‑Activated Mood Journal")
st.write("""
Record your daily journal entry using your voice.
This app uses **Whisper** to transcribe and **VADER** to analyze sentiment.
Enjoy tracking your mood over time!
""")

# Section: Upload Audio File
st.header("Upload Your Journal Entry")
uploaded_file = st.file_uploader("Choose an audio file (e.g., .wav, .mp3)", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Transcribe & Analyze"):
        with st.spinner("Processing..."):
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_filename = tmp_file.name

            # Transcribe using Whisper
            result = model.transcribe(tmp_filename)
            transcription = result["text"].strip()

            # Remove the temporary file
            os.remove(tmp_filename)

            # Perform sentiment analysis using VADER
            sentiment_scores = sia.polarity_scores(transcription)
            compound_score = sentiment_scores["compound"]

            # Create a new journal entry with a timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_entry = {"Timestamp": timestamp, "Transcription": transcription, "Sentiment": compound_score}

            # Append the new entry to the DataFrame in session_state
            st.session_state.journal_df = pd.concat(
    [st.session_state.journal_df, pd.DataFrame([new_entry])],
    ignore_index=True
)


            st.success("Transcription and sentiment analysis completed!")
            st.write("**Transcribed Entry:**")
            st.write(transcription)
            st.write("**Sentiment Score (Compound):**", compound_score)

# Section: Display Journal History & Mood Chart
st.header("Your Mood Journal Over Time")
if st.session_state.journal_df.empty:
    st.write("No entries yet. Please upload a journal entry!")
else:
    df = st.session_state.journal_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")

    st.subheader("Journal Entries")
    st.dataframe(df)

    st.subheader("Mood Over Time")
    st.line_chart(data=df.set_index("Timestamp")["Sentiment"])

# Option to clear journal entries
if st.button("Clear All Journal Entries"):
    st.session_state.journal_df = pd.DataFrame(columns=["Timestamp", "Transcription", "Sentiment"])
    st.success("Journal cleared!")
