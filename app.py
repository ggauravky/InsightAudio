import os
import time
import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input, DOWNLOAD_DIR
from core.transcriber import transcribe_all, WHISPER_MODEL
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Streamlit Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightAudio — AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Clean, Human-Centric UI Stylesheet ─────────────────────────────────────────
st.markdown("""
<style>
/* Modern, clean typography & neutral palette */
:root {
    --bg-main: #0b0f19;
    --bg-card: #111827;
    --bg-card-secondary: #1f2937;
    --border-color: #374151;
    --border-focus: #6366f1;
    --primary: #4f46e5;
    --primary-hover: #4338ca;
    --text-main: #f3f4f6;
    --text-muted: #9ca3af;
    --accent-indigo: #818cf8;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
}

/* Global Reset */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: var(--text-main);
}

.stApp {
    background-color: var(--bg-main);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border-color) !important;
}

/* Clean Header styling */
.header-container {
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1.5rem;
}

.header-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.header-subtitle {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
}

/* Content cards */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

.info-card-header {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--accent-indigo);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.metric-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
    background: var(--bg-card-secondary);
    border: 1px solid var(--border-color);
    color: var(--text-muted);
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}

.metric-pill-success {
    background: rgba(16, 185, 129, 0.1);
    border-color: rgba(16, 185, 129, 0.3);
    color: var(--accent-emerald);
}

.metric-pill-indigo {
    background: rgba(99, 102, 241, 0.1);
    border-color: rgba(99, 102, 241, 0.3);
    color: var(--accent-indigo);
}

/* Transcript Box */
.transcript-container {
    background-color: var(--bg-card-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1.2rem;
    max-height: 400px;
    overflow-y: auto;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text-main);
    white-space: pre-wrap;
    word-break: break-word;
}

/* Chat Prompt Chips */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.75rem 0 1rem 0;
}

/* Subtle scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-main);
}
::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}
</style>
""", unsafe_allow_html=True)

# ─── Initialize Session State ───────────────────────────────────────────────────
defaults = {
    "result": None,
    "chat_history": [],
    "is_processing": False,
    "current_source_name": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎙️ InsightAudio")
    st.caption("AI Meeting Assistant & Intelligence")
    st.markdown("---")

    st.markdown("#### ⚙️ Configuration")
    language = st.selectbox(
        "Speech Recognition Engine",
        options=["english", "hinglish"],
        format_func=lambda x: "English (OpenAI Whisper)" if x == "english" else "Hindi / Hinglish (Sarvam AI)",
        index=0,
        help="Select Whisper for English and global languages, or Sarvam AI for Hindi and mixed Hinglish audio.",
    )

    st.markdown("---")
    st.markdown("#### 🔑 Service Status")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    sarvam_key = os.getenv("SARVAM_API_KEY")

    if gemini_key:
        st.markdown('<span class="metric-pill metric-pill-success">● Google Gemini: Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="metric-pill" style="color:#ef4444;">● Google Gemini: Key Missing</span>', unsafe_allow_html=True)

    if sarvam_key:
        st.markdown('<span class="metric-pill metric-pill-success">● Sarvam AI: Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="metric-pill">○ Sarvam AI: Not Configured</span>', unsafe_allow_html=True)

    st.markdown(f'<span class="metric-pill metric-pill-indigo">Whisper Model: {WHISPER_MODEL}</span>', unsafe_allow_html=True)

    if st.session_state.result:
        st.markdown("---")
        if st.button("🔄 Reset / New Analysis", use_container_width=True, type="secondary"):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.session_state.current_source_name = ""
            st.rerun()

    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
        1. **Ingest**: Provide a YouTube link or upload a local audio/video file.
        2. **Process**: Audio is converted to 16kHz mono WAV and chunked.
        3. **Transcribe**: Speech is converted to text using Whisper or Sarvam AI.
        4. **Analyze**: Google Gemini synthesizes executive summary, actions, and decisions.
        5. **RAG Memory**: Transcripts are indexed in ChromaDB for instant Q&A.
        """)

# ─── Main Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-title">🎙️ InsightAudio</div>
    <div class="header-subtitle">Automated transcription, meeting intelligence, and conversational RAG querying.</div>
</div>
""", unsafe_allow_html=True)

# ─── Input & Ingestion Section ──────────────────────────────────────────────────
if not st.session_state.result:
    st.markdown("### 📥 Select Input Source")
    input_tab1, input_tab2, input_tab3 = st.tabs(["🎬 YouTube URL", "📁 Upload Media File", "💻 Local File Path"])

    input_source = ""
    source_display_name = ""

    with input_tab1:
        youtube_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any public or unlisted YouTube video URL.",
        )
        if youtube_url.strip():
            input_source = youtube_url.strip()
            source_display_name = "YouTube Video"

    with input_tab2:
        uploaded_file = st.file_uploader(
            "Upload Audio or Video File",
            type=["mp4", "mp3", "wav", "m4a", "webm", "ogg", "flac"],
            help="Drag and drop meeting recordings, lectures, or podcasts.",
        )
        if uploaded_file is not None:
            # Save uploaded file temporarily to downloads directory
            temp_path = os.path.join(DOWNLOAD_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            input_source = temp_path
            source_display_name = uploaded_file.name
            st.success(f"Loaded: `{uploaded_file.name}` ({uploaded_file.size / (1024*1024):.1f} MB)")

    with input_tab3:
        local_path = st.text_input(
            "Local File Absolute Path",
            placeholder="e.g. C:/Recordings/team_sync.mp4",
            help="Direct path for very large files already stored on your machine.",
        )
        if local_path.strip() and os.path.exists(local_path.strip()):
            input_source = local_path.strip()
            source_display_name = os.path.basename(local_path.strip())

    st.markdown("<br/>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        process_btn = st.button("⚡ Process & Analyze Meeting", type="primary", use_container_width=True)

    if process_btn:
        if not input_source:
            st.error("Please provide a valid YouTube URL, upload a file, or specify a valid local file path.")
        else:
            st.session_state.is_processing = True
            st.session_state.current_source_name = source_display_name

            with st.status("🚀 Processing meeting...", expanded=True) as status:
                try:
                    # Step 1: Audio Processing & Chunking
                    status.update(label="🔊 Step 1/6: Extracting audio and preparing 16kHz WAV chunks...", state="running")
                    chunks = process_input(input_source)
                    st.write(f"✓ Audio processed — {len(chunks)} chunk(s) prepared.")

                    # Step 2: Speech-to-Text Transcription
                    engine_name = "Sarvam AI (Indic)" if language == "hinglish" else f"Whisper ({WHISPER_MODEL})"
                    status.update(label=f"🎙️ Step 2/6: Transcribing speech with {engine_name}...", state="running")
                    transcript = transcribe_all(chunks, language=language)
                    st.write(f"✓ Transcription complete ({len(transcript.split())} words).")

                    # Step 3: Title Generation
                    status.update(label="🏷️ Step 3/6: Generating session title...", state="running")
                    title = generate_title(transcript)
                    st.write(f"✓ Title: **{title}**")

                    # Step 4: Summarization
                    status.update(label="📋 Step 4/6: Synthesizing executive summary...", state="running")
                    summary = summarize(transcript)
                    st.write("✓ Executive summary synthesized.")

                    # Step 5: Action Items & Decisions
                    status.update(label="🔍 Step 5/6: Extracting action items and key decisions...", state="running")
                    action_items = extract_action_items(transcript)
                    decisions = extract_key_decisions(transcript)
                    questions = extract_questions(transcript)
                    st.write("✓ Extracted deliverables, decisions, and open questions.")

                    # Step 6: Vector Indexing for RAG
                    status.update(label="🧠 Step 6/6: Indexing transcript in ChromaDB vector store...", state="running")
                    rag_chain = build_rag_chain(transcript)
                    st.write("✓ RAG vector memory ready for conversational querying.")

                    # Finalize session
                    st.session_state.result = {
                        "title": title,
                        "transcript": transcript,
                        "summary": summary,
                        "action_items": action_items,
                        "key_decisions": decisions,
                        "open_questions": questions,
                        "rag_chain": rag_chain,
                        "chunk_count": len(chunks),
                        "word_count": len(transcript.split()),
                        "language": language,
                        "source": source_display_name,
                    }
                    st.session_state.chat_history = []
                    status.update(label="✅ Meeting analysis complete!", state="complete", expanded=False)
                    time.sleep(0.5)
                    st.rerun()

                except Exception as e:
                    status.update(label=f"❌ Error during processing: {str(e)}", state="error")
                    st.error(f"Error details: {e}")

# ─── Results Dashboard ──────────────────────────────────────────────────────────
if st.session_state.result:
    res = st.session_state.result

    # Session Info Banner
    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-header">📌 Session Overview</div>
        <h2 style="margin: 0 0 0.5rem 0; font-size: 1.45rem; font-weight: 700; color: #ffffff;">{res['title']}</h2>
        <div>
            <span class="metric-pill metric-pill-indigo">📄 {res['word_count']} words</span>
            <span class="metric-pill metric-pill-indigo">🧩 {res['chunk_count']} audio chunk(s)</span>
            <span class="metric-pill metric-pill-success">🌐 {res['language'].capitalize()}</span>
            <span class="metric-pill">📁 Source: {res['source']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dashboard Tabs
    tab_summary, tab_actions, tab_questions, tab_chat, tab_transcript, tab_export = st.tabs([
        "📋 Executive Summary",
        "✅ Actions & Decisions",
        "❓ Questions & Risks",
        "💬 Meeting Chat (RAG)",
        "📝 Full Transcript",
        "📥 Export",
    ])

    # ── Tab 1: Executive Summary ───────────────────────────────────────────────
    with tab_summary:
        st.markdown("#### 📋 Executive Summary")
        st.markdown(res["summary"])

    # ── Tab 2: Actions & Decisions ─────────────────────────────────────────────
    with tab_actions:
        col_act, col_dec = st.columns(2, gap="large")
        with col_act:
            st.markdown("#### ✅ Action Items & Tasks")
            st.markdown(res["action_items"])
        with col_dec:
            st.markdown("#### 🔑 Key Decisions Log")
            st.markdown(res["key_decisions"])

    # ── Tab 3: Questions & Follow-ups ──────────────────────────────────────────
    with tab_questions:
        st.markdown("#### ❓ Open Questions & Follow-Up Items")
        st.markdown(res["open_questions"])

    # ── Tab 4: Interactive RAG Chat ────────────────────────────────────────────
    with tab_chat:
        st.markdown("#### 💬 Ask Questions About This Meeting")
        st.caption("Answers are retrieved directly from the indexed transcript using ChromaDB and Google Gemini.")

        # Quick starter prompt suggestions
        st.markdown("**Suggested Questions:**")
        col_p1, col_p2, col_p3 = st.columns(3)
        prompt_to_submit = None

        with col_p1:
            if st.button("📌 What were the key decisions?", use_container_width=True):
                prompt_to_submit = "What were the key decisions made in this meeting?"
        with col_p2:
            if st.button("📋 Who has assigned action items?", use_container_width=True):
                prompt_to_submit = "List all action items and who is responsible for each."
        with col_p3:
            if st.button("❓ What topics were unresolved?", use_container_width=True):
                prompt_to_submit = "What questions or issues remained unresolved at the end of the meeting?"

        # Display Chat History using native Streamlit chat_message
        st.markdown("---")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle Chat Input
        user_query = st.chat_input("Ask a question about this meeting...")
        active_query = prompt_to_submit or user_query

        if active_query:
            st.session_state.chat_history.append({"role": "user", "content": active_query})
            with st.chat_message("user"):
                st.markdown(active_query)

            with st.chat_message("assistant"):
                with st.spinner("Searching transcript context..."):
                    answer = ask_question(res["rag_chain"], active_query)
                    st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state.chat_history:
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

    # ── Tab 5: Full Transcript ─────────────────────────────────────────────────
    with tab_transcript:
        st.markdown("#### 📝 Full Meeting Transcript")
        search_query = st.text_input("🔍 Search transcript", placeholder="Filter by keyword or phrase...")

        if search_query.strip():
            matched_lines = [
                line for line in res["transcript"].split("\n")
                if search_query.lower() in line.lower()
            ]
            st.caption(f"Found {len(matched_lines)} matching segment(s):")
            st.markdown(
                f'<div class="transcript-container">{"\n".join(matched_lines)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="transcript-container">{res["transcript"]}</div>',
                unsafe_allow_html=True,
            )

    # ── Tab 6: Export ──────────────────────────────────────────────────────────
    with tab_export:
        st.markdown("#### 📥 Export Meeting Documentation")
        st.write("Download the generated meeting intelligence report and transcript for archiving or distribution.")

        # Prepare Markdown Report content
        report_md = f"""# {res['title']}
*Generated by InsightAudio*

---

## 📋 Executive Summary
{res['summary']}

---

## ✅ Action Items
{res['action_items']}

---

## 🔑 Key Decisions
{res['key_decisions']}

---

## ❓ Open Questions
{res['open_questions']}

---

## 📝 Full Transcript
{res['transcript']}
"""

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.download_button(
                label="📄 Download Full Meeting Report (.md)",
                data=report_md,
                file_name=f"{res['title'].replace(' ', '_').lower()}_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with c_exp2:
            st.download_button(
                label="📝 Download Raw Transcript (.txt)",
                data=res["transcript"],
                file_name=f"{res['title'].replace(' ', '_').lower()}_transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )