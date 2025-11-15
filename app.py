import streamlit as st
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


# --- Helper Function to Get Video ID from URL ---
def extract_video_id(link):
    """
    Extract the video id from a YouTube video link.
    """
    # Parse the link using urlparse
    parsed_url = urlparse(link)

    if parsed_url.netloc == "www.youtube.com":
        # Extract the video id from the query parameters for the www.youtube.com format
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:
            return query_params["v"][0]
        else:
            return None
    elif parsed_url.netloc == "youtu.be":
        # Extract the video id from the path for the youtu.be format
        path = parsed_url.path
        if path.startswith("/"):
            path = path[1:]
        return path
    else:
        # Return None for all other link formats
        return None


def transcript_snippets_to_string(snippets, sep=" "):
    """
    Convert a list of transcript snippets to one plain string.
    Accepts items like {'text':..., 'start':...} or objects with .text/.start.
    """
    import re

    def get_start(item):
        return getattr(
            item, "start", item.get("start", 0) if isinstance(item, dict) else 0
        )

    def get_text(item):
        return getattr(
            item, "text", item.get("text", "") if isinstance(item, dict) else ""
        )

    # sort by start time, extract and strip texts
    texts = [get_text(s).strip() for s in sorted(snippets, key=get_start)]
    # join, remove spaces before punctuation and collapse multiple spaces
    joined = sep.join(t for t in texts if t)
    joined = re.sub(r"\s+([.,!?;:])", r"\1", joined)
    joined = re.sub(r"\s{2,}", " ", joined).strip()
    return joined


# --- Helper Function to Get Transcript ---
@st.cache_data
def get_transcript(video_id):
    """Fetches the transcript for a given video ID."""
    try:
        # Create an instance of the API
        api = YouTubeTranscriptApi()

        # Now call the list method
        transcript_list = api.list(video_id=video_id)

        # You can then find and fetch a transcript
        transcript = transcript_list.find_transcript(["en"])

        snippets = transcript.fetch()  # or transcript_list.find_transcript(...).fetch()
        full_text = transcript_snippets_to_string(snippets)
        return full_text

    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        st.error(
            "Could not retrieve transcript. This might be because transcripts are disabled for this video or the video is invalid."
        )
        return None


# --- Helper Function to Get Summary from Gemini ---
def get_summary_from_gemini(transcript):
    """Generates a summary from the transcript using the Gemini API."""
    try:
        # Initialize the Gemini model
        # Using 1.5-pro as it's broadly available and handles long contexts
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
You are a YouTube video summarizer. 
Your task is to provide a concise, easy-to-read summary of the following video transcript. 
Focus on the main points and key takeaways. Use bullet points if it helps clarity. And if
required use your general knowledge to answer the questions.

Transcript:
{transcript}

Summary:
"""
        # Generate the content
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None


# --- Initialize Session State ---
def initialize_session_state():
    """Initialize session state variables for chat."""
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "transcript" not in st.session_state:
        st.session_state.transcript = None
    if "video_loaded" not in st.session_state:
        st.session_state.video_loaded = False
    if "summary" not in st.session_state:
        st.session_state.summary = None


# --- Streamlit UI ---
st.set_page_config(
    page_title="YouTube Summarizer & Chat", page_icon="🚀", layout="wide"
)
st.title("🚀 YouTube Summarizer & Chat Assistant")
st.write("Paste a YouTube URL, get a summary, and chat about the video!")

# Load API key from environment
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ API key not found. Please add GOOGLE_API_KEY to your .env file.")
else:
    # Configure the GenAI library
    genai.configure(api_key=api_key)

    # Initialize session state
    initialize_session_state()

    # --- Main Page ---
    col1, col2 = st.columns([2, 1])

    with col1:
        video_url = st.text_input("Enter the YouTube video URL:")

    with col2:
        if st.button("📊 Summarize Video", use_container_width=True):
            # 1. Validate inputs
            if not video_url:
                st.error("Please enter a YouTube video URL.")
            else:
                with st.spinner(
                    "Working on it... Fetching transcript and summarizing..."
                ):
                    try:
                        # 2. Get Video ID
                        video_id = extract_video_id(video_url)

                        if not video_id:
                            st.error("Invalid YouTube URL. Please check and try again.")
                        else:
                            # 3. Get Transcript
                            transcript = get_transcript(video_id)

                            if transcript:
                                # 4. Get Summary
                                st.info(
                                    "Transcript found! Now summarizing with Gemini..."
                                )
                                summary = get_summary_from_gemini(transcript)

                                if summary:
                                    # Store in session state
                                    st.session_state.transcript = transcript
                                    st.session_state.summary = summary
                                    st.session_state.video_loaded = True
                                    st.session_state.messages = []

                                    # Initialize chat session

                                    system_prompt = f"""You are a helpful YouTube video assistant. 
You have access to the following video transcript and should answer user questions about it.

VIDEO TRANSCRIPT:
{transcript}

VIDEO SUMMARY:
{summary}

Please answer questions about this video content accurately and helpfully. 
If a question is not related to the video, politely redirect the conversation back to the video.
Use your general knowledge to provide better context when needed."""
                                    model = genai.GenerativeModel(
                                        "gemini-2.5-flash",
                                        system_instruction=system_prompt,
                                    )

                                    st.session_state.chat_session = model.start_chat(
                                        history=[]
                                    )

                                    st.success(
                                        "✅ Video loaded! You can now view the summary and chat about it."
                                    )

                    except Exception as e:
                        # Catch-all for any unexpected errors
                        st.error(f"An unexpected error occurred: {e}")

    # --- Display Summary and Chat ---
    if st.session_state.video_loaded:
        st.divider()

        # Create tabs for Summary and Chat
        tab1, tab2 = st.tabs(["📋 Summary", "💬 Chat"])

        with tab1:
            st.subheader("Video Summary")
            st.markdown(st.session_state.summary)

        with tab2:
            st.subheader("Chat with Video Assistant")

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            user_input = st.chat_input("Ask a question about the video...")

            if user_input:
                # Add user message to history
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )

                # Get AI response
                try:
                    with st.spinner("AI is thinking..."):
                        response = st.session_state.chat_session.send_message(
                            user_input
                        )
                        assistant_message = response.text

                    # Add assistant message to history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": assistant_message}
                    )

                    # Rerun to display all messages including the new ones
                    st.rerun()

                except Exception as e:
                    st.error(f"Error getting response: {e}")
    else:
        st.info("📹 Load a YouTube video to view its summary and start chatting!")
