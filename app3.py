
import streamlit as st
import hashlib
import time
import random
import json
import re
from datetime import datetime
from io import BytesIO
from urllib.parse import urlparse, parse_qs

from gtts import gTTS
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart YouTube Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Smart Interactive YouTube Lecture Assistant")
st.caption(
    "YouTube transcript • AI Q&A • Summary • Interactive Quiz "
    "• Audio answers • Quiz Report"
)

PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.8-flash"


# =========================================================
# EMBEDDINGS
# =========================================================

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# FETCH YOUTUBE TRANSCRIPT
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_transcript(video_id):
    api = YouTubeTranscriptApi()

    fetched = api.fetch(
        video_id,
        languages=["en", "hi"]
    )

    return [
        {
            "text": item.text,
            "start": float(item.start)
        }
        for item in fetched
    ]


# =========================================================
# TEXT TO SPEECH
# =========================================================

@st.cache_data(show_spinner=False)
def create_audio(text):
    output = BytesIO()

    gTTS(
        text=text[:4000],
        lang="en"
    ).write_to_fp(output)

    return output.getvalue()


# =========================================================
# GEMINI RETRY AND FALLBACK
# =========================================================

def is_temporary_error(exc):
    message = str(exc).lower()

    return any(
        marker in message
        for marker in [
            "503",
            "429",
            "500",
            "502",
            "504",
            "unavailable",
            "resource_exhausted",
            "internal server error",
            "deadline exceeded",
            "timed out"
        ]
    )


def generate_with_fallback(
    prompt,
    api_key,
    max_output_tokens=1024
):
    errors = []

    for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
            max_output_tokens=max_output_tokens,
            max_retries=0
        )

        for attempt in range(3):
            try:
                result = (llm | StrOutputParser()).invoke(prompt)
                return result, model_name

            except Exception as exc:
                errors.append(
                    f"{model_name}: {str(exc)[:250]}"
                )

                if not is_temporary_error(exc):
                    break

                if attempt < 2:
                    time.sleep(
                        (1.5 * (2 ** attempt))
                        + random.uniform(0, 0.5)
                    )

    raise RuntimeError(
        "Both Gemini models failed. Check model availability, "
        "API key, quota, or try again later.\n\n"
        + "\n".join(errors[-4:])
    )


# =========================================================
# EXTRACT YOUTUBE VIDEO ID
# =========================================================

def extract_video_id(url):
    try:
        parsed = urlparse(url.strip())
        hostname = parsed.hostname or ""

        if hostname in ("youtu.be", "www.youtu.be"):
            return parsed.path.strip("/").split("/")[0] or None

        if hostname.endswith("youtube.com"):
            if parsed.path == "/watch":
                return parse_qs(parsed.query).get("v", [None])[0]

            if parsed.path.startswith(("/embed/", "/shorts/")):
                parts = parsed.path.strip("/").split("/")
                return parts[1] if len(parts) > 1 else None

    except Exception:
        pass

    return None


# =========================================================
# BUILD DOCUMENTS FOR RAG
# =========================================================

def build_documents(text, video_id, segments=None):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80
    )

    if segments:
        docs = []
        current_text = ""
        current_start = 0.0

        for segment in segments:
            part = segment["text"].strip()

            if not part:
                continue

            if current_text and len(current_text) + len(part) > 700:
                docs.append(
                    Document(
                        page_content=current_text.strip(),
                        metadata={
                            "video_id": video_id,
                            "start": current_start
                        }
                    )
                )

                current_text = ""

            if not current_text:
                current_start = segment["start"]

            current_text += " " + part

        if current_text.strip():
            docs.append(
                Document(
                    page_content=current_text.strip(),
                    metadata={
                        "video_id": video_id,
                        "start": current_start
                    }
                )
            )

        return docs

    return [
        Document(
            page_content=chunk,
            metadata={
                "video_id": video_id,
                "start": 0.0
            }
        )
        for chunk in splitter.split_text(text)
    ]


# =========================================================
# INDEX TRANSCRIPT
# =========================================================

def index_transcript(text, video_id, segments=None):
    if not text.strip():
        st.error("Transcript is empty.")
        return False

    fingerprint = hashlib.sha256(
        (video_id + text).encode("utf-8")
    ).hexdigest()

    if st.session_state.get("source_hash") == fingerprint:
        st.success(
            "Transcript already processed. Reusing search index."
        )
        return True

    with st.spinner("Building search index... first run may take longer"):
        docs = build_documents(text, video_id, segments)

        if not docs:
            st.error("No usable transcript chunks were found.")
            return False

        store = FAISS.from_documents(
            docs,
            get_embeddings()
        )

    st.session_state.video_id = video_id
    st.session_state.full_text = text
    st.session_state.vectorstore = store
    st.session_state.source_hash = fingerprint

    st.session_state.summary = None
    st.session_state.summary_model = None

    st.session_state.quiz = None
    st.session_state.quiz_model = None
    st.session_state.quiz_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.quiz_report = None

    st.session_state.answer = None
    st.session_state.answer_audio = None
    st.session_state.answer_model = None
    st.session_state.answer_docs = []

    for i in range(3):
        st.session_state.pop(f"quiz_choice_{i}", None)

    st.success(
        f"Transcript ready! {len(docs)} chunks indexed."
    )

    return True


# =========================================================
# PARSE AND VALIDATE QUIZ JSON
# =========================================================

def parse_quiz_json(raw_text):
    cleaned = raw_text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Gemini did not return a JSON quiz. Please generate again."
        )

    data = json.loads(cleaned[start:end + 1])
    questions = data.get("questions")

    if not isinstance(questions, list) or len(questions) != 3:
        raise ValueError("Quiz must contain exactly 3 questions.")

    validated = []

    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Invalid question format.")

        question = str(item.get("question", "")).strip()
        options = item.get("options", [])
        answer = str(item.get("answer", "")).strip().upper()
        explanation = str(item.get("explanation", "")).strip()

        if (
            not question
            or not isinstance(options, list)
            or len(options) != 4
        ):
            raise ValueError(
                "Each question must have a question and four options."
            )

        if answer not in ("A", "B", "C", "D"):
            raise ValueError(
                "Each correct answer must be labelled A, B, C, or D."
            )

        validated.append({
            "question": question,
            "options": [str(option).strip() for option in options],
            "answer": answer,
            "explanation": explanation or "No explanation was provided."
        })

    return {"questions": validated}


# =========================================================
# QUIZ REPORT HELPERS
# =========================================================

def option_text(question, letter):
    if letter not in ("A", "B", "C", "D"):
        return "Not answered"

    index = ord(letter) - ord("A")

    return f"{letter}) {question['options'][index]}"


def create_quiz_report(
    questions,
    user_answers,
    video_id,
    model_name
):
    total = len(questions)

    correct = sum(
        user_answers.get(i) == q["answer"]
        for i, q in enumerate(questions)
    )

    incorrect = total - correct

    percentage = (
        (correct / total) * 100
        if total else 0
    )

    lines = [
        "SMART INTERACTIVE YOUTUBE LECTURE ASSISTANT",
        "QUIZ PERFORMANCE REPORT",
        "=" * 48,
        f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        f"YouTube Video ID: {video_id or 'Not available'}",
        f"Quiz model: {model_name or 'Not available'}",
        "",
        "FINAL RESULT",
        f"Total questions: {total}",
        f"Correct answers: {correct}",
        f"Incorrect or unanswered: {incorrect}",
        f"Score: {correct}/{total}",
        f"Percentage: {percentage:.2f}%",
        "",
        "DETAILED ANSWERS",
        "=" * 48
    ]

    for i, q in enumerate(questions):
        chosen = user_answers.get(i)

        lines.extend([
            "",
            f"Question {i + 1}: {q['question']}",
            f"Your answer: {option_text(q, chosen)}",
            f"Correct answer: {option_text(q, q['answer'])}",
            f"Result: {'CORRECT' if chosen == q['answer'] else 'INCORRECT'}",
            f"Explanation: {q['explanation']}",
            "-" * 40
        ])

    return "\n".join(lines), correct, incorrect, percentage


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "video_id": None,
    "full_text": "",
    "vectorstore": None,
    "source_hash": None,
    "summary": None,
    "quiz": None,
    "answer": None,
    "answer_audio": None,
    "quiz_answers": {},
    "quiz_submitted": False,
    "quiz_report": None,
    "quiz_model": None,
    "answer_model": None,
    "answer_docs": [],
    "summary_model": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SIDEBAR
# =========================================================

api_key = st.sidebar.text_input(
    "Gemini API key",
    type="password"
)

st.sidebar.caption(
    "Your key is used for requests from this app. "
    "Do not publish it in your code or GitHub repository."
)

st.sidebar.caption(f"Primary model: {PRIMARY_MODEL}")
st.sidebar.caption(f"Fallback model: {FALLBACK_MODEL}")


# =========================================================
# VIDEO INPUT
# =========================================================

url = st.text_input(
    "YouTube video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

manual_text = st.text_area(
    "Manual transcript fallback",
    placeholder="Paste transcript text here if YouTube fetching fails.",
    height=120
)

col1, col2 = st.columns(2)

fetch_clicked = col1.button(
    "Fetch transcript",
    use_container_width=True
)

manual_clicked = col2.button(
    "Process pasted transcript",
    use_container_width=True
)

video_id = extract_video_id(url) if url else None


# =========================================================
# FETCH TRANSCRIPT
# =========================================================

if fetch_clicked:
    if not video_id:
        st.error("Enter a valid YouTube URL.")
    else:
        try:
            with st.spinner("Fetching transcript from YouTube..."):
                segments = fetch_transcript(video_id)

            full_text = " ".join(
                item["text"] for item in segments
            ).strip()

            if full_text:
                index_transcript(
                    full_text,
                    video_id,
                    segments
                )
            else:
                st.warning("No transcript text was returned.")

        except Exception as exc:
            st.error(
                "Automatic fetching failed. Paste the transcript "
                "manually and click 'Process pasted transcript'."
            )

            with st.expander("Technical details"):
                st.code(str(exc)[:1500])


# =========================================================
# MANUAL TRANSCRIPT
# =========================================================

if manual_clicked:
    if not video_id:
        st.error("Enter a valid YouTube URL first.")

    elif not manual_text.strip():
        st.error("Paste transcript text first.")

    else:
        index_transcript(
            manual_text.strip(),
            video_id
        )


# =========================================================
# MAIN FEATURES
# =========================================================

if st.session_state.vectorstore is not None:

    # =====================================================
    # Q&A — ALWAYS ANSWER IN ENGLISH
    # =====================================================

    st.divider()
    st.subheader("💬 Ask a question")

    question = st.text_input(
        "What do you want to understand?",
        key="question_input"
    )

    if st.button("Generate answer", type="primary"):

        if not api_key:
            st.warning(
                "Enter your Gemini API key in the sidebar."
            )

        elif not question.strip():
            st.warning("Enter a question.")

        else:
            try:
                with st.spinner(
                    "Searching and generating English answer..."
                ):
                    relevant_docs = (
                        st.session_state.vectorstore
                        .similarity_search(question, k=2)
                    )

                    context = "\n\n".join(
                        doc.page_content
                        for doc in relevant_docs
                    )

                    prompt = ChatPromptTemplate.from_template(
                        """You are an English-language lecture assistant.

STRICT LANGUAGE RULES:
- Always write your entire answer in English.
- Never answer in Hindi or Hinglish.
- If the lecture context is in Hindi, understand it and
  translate the relevant information into natural English.
- Use simple, clear English that students can understand.
- Use only the information provided in the lecture context.
- If the answer is not available in the context, say:
  "This information is not available in the provided lecture transcript."
- Do not invent facts.
- Keep the answer concise and explain technical concepts clearly.

Lecture context:
{context}

Student's question:
{question}

Answer in English:"""
                    )

                    formatted_prompt = prompt.format(
                        context=context,
                        question=question
                    )

                    answer, model_used = generate_with_fallback(
                        formatted_prompt,
                        api_key
                    )

                st.session_state.answer = answer
                st.session_state.answer_audio = None
                st.session_state.answer_model = model_used
                st.session_state.answer_docs = relevant_docs

            except Exception as exc:
                st.error("Answer generation failed.")

                with st.expander("Technical details"):
                    st.code(str(exc)[:2000])

    if st.session_state.answer:

        st.markdown("### Answer")
        st.write(st.session_state.answer)

        st.caption(
            "Generated using "
            f"{st.session_state.get('answer_model') or PRIMARY_MODEL}"
        )

        st.markdown("**Relevant lecture timestamps**")

        for doc in st.session_state.get("answer_docs", []):
            start = int(doc.metadata.get("start", 0))
            mins, secs = divmod(start, 60)

            timestamp_url = (
                f"https://www.youtube.com/watch?v="
                f"{st.session_state.video_id}&t={start}s"
            )

            st.markdown(
                f"- [{mins:02d}:{secs:02d}]({timestamp_url})"
            )

        if st.button("Create spoken answer"):
            try:
                with st.spinner("Creating audio..."):
                    st.session_state.answer_audio = create_audio(
                        st.session_state.answer
                    )

            except Exception as exc:
                st.error(f"Audio generation failed: {exc}")

        if st.session_state.answer_audio:
            st.audio(
                st.session_state.answer_audio,
                format="audio/mp3"
            )

    # =====================================================
    # LECTURE SUMMARY
    # =====================================================

    st.divider()
    st.subheader("📝 Lecture summary")

    if st.button("Generate summary"):

        if not api_key:
            st.warning("Enter your Gemini API key.")

        else:
            try:
                summary_prompt = ChatPromptTemplate.from_template(
                    """Summarize the lecture in English using headings
and bullet points.

Requirements:
- Write the entire summary in English.
- If the transcript is in Hindi, translate the key information
  into natural English.
- Include the main concepts and important takeaways.
- Use simple, student-friendly language.
- Do not invent facts.

Transcript:
{transcript}

English summary:"""
                )

                with st.spinner("Generating English summary..."):
                    summary, model_used = generate_with_fallback(
                        summary_prompt.format(
                            transcript=st.session_state.full_text[:12000]
                        ),
                        api_key,
                        max_output_tokens=2048
                    )

                st.session_state.summary = summary
                st.session_state.summary_model = model_used

            except Exception as exc:
                st.error("Summary generation failed.")

                with st.expander("Technical details"):
                    st.code(str(exc)[:1500])

    if st.session_state.summary:
        st.markdown(st.session_state.summary)

        st.caption(
            f"Model: {st.session_state.get('summary_model') or PRIMARY_MODEL}"
        )

    # =====================================================
    # INTERACTIVE QUIZ
    # =====================================================

    st.divider()
    st.subheader("🧠 Interactive lecture quiz")

    if st.button(
        "Generate new English quiz",
        type="primary"
    ):
        if not api_key:
            st.warning("Enter your Gemini API key.")

        else:
            try:
                quiz_prompt = ChatPromptTemplate.from_template(
                    """Create exactly 3 multiple-choice questions
from the lecture transcript.

Requirements:
- Every question must be in English.
- Every option must be in English.
- Every explanation must be in English.
- Each question must have exactly four options.
- There must be exactly one correct answer per question.
- Use answer labels A, B, C, or D.
- Use only information from the transcript.
- Do not invent facts.
- Return ONLY valid JSON without Markdown fences.

Required structure:
{{
  "questions": [
    {{
      "question": "What does LAN stand for?",
      "options": [
        "Local Area Network",
        "Large Access Network",
        "Long Area Node",
        "Local Access Node"
      ],
      "answer": "A",
      "explanation": "LAN stands for Local Area Network."
    }}
  ]
}}

Transcript:
{transcript}

Return the complete quiz JSON in English."""
                )

                with st.spinner("Generating your English quiz..."):
                    raw_quiz, model_used = generate_with_fallback(
                        quiz_prompt.format(
                            transcript=st.session_state.full_text[:12000]
                        ),
                        api_key,
                        max_output_tokens=1800
                    )

                new_quiz = parse_quiz_json(raw_quiz)

                st.session_state.quiz = new_quiz
                st.session_state.quiz_model = model_used
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_report = None

                for i in range(3):
                    st.session_state.pop(f"quiz_choice_{i}", None)

                st.success("New English quiz generated!")

            except Exception as exc:
                st.error("Quiz generation failed.")

                with st.expander("Technical details"):
                    st.code(str(exc)[:2000])

    # =====================================================
    # STUDENT ANSWERS AND SUBMISSION
    # =====================================================

    quiz_data = st.session_state.quiz

    if quiz_data:
        questions = quiz_data["questions"]

        st.caption(
            "Quiz generated using "
            f"{st.session_state.get('quiz_model') or PRIMARY_MODEL}"
        )

        if not st.session_state.quiz_submitted:

            st.write(
                "Choose one answer for every question, "
                "then submit to see your result."
            )

            with st.form("interactive_quiz_form"):

                submitted_answers = {}

                for i, q in enumerate(questions):

                    st.markdown(
                        f"**Question {i + 1}. {q['question']}**"
                    )

                    submitted_answers[i] = st.radio(
                        f"Choose an answer for question {i + 1}:",
                        options=["A", "B", "C", "D"],
                        format_func=lambda letter, opts=q["options"]: (
                            f"{letter}) "
                            f"{opts[ord(letter) - ord('A')]}"
                        ),
                        index=None,
                        key=f"quiz_choice_{i}"
                    )

                submit_quiz = st.form_submit_button(
                    "Submit quiz",
                    type="primary"
                )

            if submit_quiz:

                unanswered = [
                    i + 1
                    for i, ans in submitted_answers.items()
                    if ans is None
                ]

                if unanswered:
                    st.warning(
                        "Please answer every question before submitting. "
                        "Unanswered question(s): "
                        + ", ".join(map(str, unanswered))
                        + "."
                    )

                else:
                    # Save the student's actual answers.
                    st.session_state.quiz_answers = submitted_answers
                    st.session_state.quiz_submitted = True

                    # Generate and save the final report ONCE.
                    report, correct, incorrect, percentage = (
                        create_quiz_report(
                            questions,
                            submitted_answers,
                            st.session_state.video_id,
                            st.session_state.quiz_model
                        )
                    )

                    st.session_state.quiz_report = report

                    # Rerun so the submitted quiz and feedback display
                    # consistently with the saved answers.
                    st.rerun()

        # =================================================
        # FEEDBACK AND FINAL REPORT AFTER SUBMISSION
        # =================================================

        if (
            st.session_state.quiz_submitted
            and st.session_state.quiz_report
        ):
            report = st.session_state.quiz_report
            user_answers = st.session_state.quiz_answers

            total = len(questions)

            correct = sum(
                user_answers.get(i) == q["answer"]
                for i, q in enumerate(questions)
            )

            incorrect = total - correct

            percentage = (
                correct / total * 100
                if total else 0
            )

            st.divider()
            st.subheader("📊 Quiz performance report")

            metric1, metric2, metric3, metric4 = st.columns(4)

            metric1.metric("Total questions", total)
            metric2.metric("Correct", correct)
            metric3.metric("Incorrect", incorrect)
            metric4.metric("Final score", f"{percentage:.1f}%")

            st.progress(percentage / 100)

            if percentage == 100:
                st.success(
                    "Excellent! You answered every question correctly."
                )

            elif percentage >= 60:
                st.success(
                    "Good work! Review the explanations to improve further."
                )

            else:
                st.info(
                    "Review the explanations and try another quiz to improve."
                )

            # ---------------------------------------------
            # ANSWER-BY-ANSWER FEEDBACK
            # ---------------------------------------------

            st.markdown("### Answer-by-answer feedback")

            for i, q in enumerate(questions):

                chosen = user_answers.get(i)
                is_correct = chosen == q["answer"]

                with st.container(border=True):

                    st.markdown(
                        f"**Question {i + 1}: {q['question']}**"
                    )

                    st.write(
                        f"Your answer: {option_text(q, chosen)}"
                    )

                    st.write(
                        f"Correct answer: "
                        f"{option_text(q, q['answer'])}"
                    )

                    if is_correct:
                        st.success("Correct!")

                    else:
                        st.error(
                            "Incorrect. Review the correct answer below."
                        )

                    st.write(
                        f"Explanation: {q['explanation']}"
                    )

            # ---------------------------------------------
            # DOWNLOAD FINAL REPORT
            # ---------------------------------------------

            st.divider()
            st.markdown("### 📄 Download your final report")

            st.download_button(
                label="⬇️ Download detailed quiz report (.txt)",
                data=report,
                file_name=(
                    "quiz_report_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                ),
                mime="text/plain",
                use_container_width=True
            )

            with st.expander("View complete text report"):
                st.text(report)

else:
    st.info(
        "Fetch a YouTube transcript or paste transcript text to begin."
    )
