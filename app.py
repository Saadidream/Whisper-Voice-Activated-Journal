import streamlit as st
import whisper
import nltk

# Check if the VADER lexicon is installed; if not, download it.
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tempfile
import os
import pandas as pd
import datetime

# Cache the Whisper model to avoid reloading on every interaction
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_whisper_model():
    return whisper.load_model("base")

# Load the model once
model = load_whisper_model()

# Initialize or retrieve the journal DataFrame from Streamlit session_state
if "journal_df" not in st.session_state:
    st.session_state.journal_df = pd.DataFrame(columns=["Timestamp", "Transcription", "Sentiment"])

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# App title and description
st.title("Voice‑Activated Mood Journal")
st.write("""
Record your daily journal entry using your voice! This app transcribes your audio with **OpenAI's Whisper** 
and performs sentiment analysis with **VADER** to track your mood over time.
""")

# Section: Upload Audio
st.header("Upload Your Journal Entry")
uploaded_file = st.file_uploader("Choose an audio file (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # Optionally preview the audio file
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Transcribe & Analyze"):
        with st.spinner("Processing..."):
            # Save the uploaded file to a temporary file
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_filename = tmp_file.name

            # Transcribe using Whisper
            result = model.transcribe(tmp_filename)
            transcription = result["text"].strip()

            # Remove the temp file
            os.remove(tmp_filename)

            # Perform sentiment analysis with VADER
            scores = sia.polarity_scores(transcription)
            compound_score = scores["compound"]

            # Create a new journal entry
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_entry = {
                "Timestamp": timestamp,
                "Transcription": transcription,
                "Sentiment": compound_score
            }

            # Append entry to the session_state DataFrame
            st.session_state.journal_df = st.session_state.journal_df.append(new_entry, ignore_index=True)

            # Display results
            st.success("Transcription and sentiment analysis completed!")
            st.write("**Transcribed Entry:**")
            st.write(transcription)
            st.write("**Sentiment Score (Compound):**", compound_score)

# Section: Mood Journal Dashboard
st.header("Your Mood Journal Over Time")

# Check if we have any entries
if st.session_state.journal_df.empty:
    st.write("No entries yet. Please upload a journal entry!")
else:
    df = st.session_state.journal_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")

    # Display entries in a table
    st.subheader("Journal Entries")
    st.dataframe(df)

    # Plot sentiment over time
    st.subheader("Mood Trend")
    st.line_chart(data=df.set_index("Timestamp")["Sentiment"])

# Button to clear the journal
if st.button("Clear All Journal Entries"):
    st.session_state.journal_df = pd.DataFrame(columns=["Timestamp", "Transcription", "Sentiment"])
    st.success("Journal cleared!")
