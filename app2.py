import os
import re
import json
import base64
import io
import streamlit as st
from gtts import gTTS
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# 1. Page Configuration
st.set_page_config(page_title="Smart Interactive YouTube QA Bot", layout="wide")
st.title("Smart Interactive YouTube Video Q&A Assistant ")

# 2. API Key Setup
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

def extract_video_id(user_input):
    user_input = user_input.strip()
    if len(user_input) == 11 and re.match(r"^[0-9A-Za-z_-]{11}$", user_input):
        return user_input
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, user_input)
    return match.group(1) if match else None

def format_seconds_to_time(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"

st.subheader("Step 1: Input YouTube Video")
video_input = st.text_input("Enter YouTube Video URL or Video ID:")

if st.button("Process Video"):
    if not api_key:
        st.error("first enter your Gemini API Key in the sidebar.")
    elif not video_input:
        st.warning("please enter YouTube Video ID .")
    else:
        video_id = extract_video_id(video_input)
        if not video_id:
            st.error("Invalid youtube input! Could not extract Video ID.")
        else:
            with st.spinner("processing video"):
                try:
                    ytt_api = YouTubeTranscriptApi()
                    fetched = ytt_api.fetch(video_id, languages=["en", "hi"])
                    
                    docs = []
                    full_text = ""
                    current_chunk_text = ""
                    chunk_start_time = 0
                    
                    for snippet in fetched:
                        if not current_chunk_text:
                            chunk_start_time = snippet.start
                        
                        text_snippet = snippet.text
                        current_chunk_text += " " + text_snippet
                        full_text += " " + text_snippet
                        
                        if len(current_chunk_text) > 1000:
                            docs.append(Document(
                                page_content=current_chunk_text,
                                metadata={"source": video_id, "start_time": chunk_start_time, "time_formatted": format_seconds_to_time(chunk_start_time)}
                            ))
                            current_chunk_text = ""
                    
                    if current_chunk_text:
                        docs.append(Document(
                            page_content=current_chunk_text,
                            metadata={"source": video_id, "start_time": chunk_start_time, "time_formatted": format_seconds_to_time(chunk_start_time)}
                        ))

                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectorstore = FAISS.from_documents(docs, embeddings)

                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["video_id"] = video_id
                    st.session_state["full_text"] = full_text
                    st.session_state.pop("quiz_data", None)
                    st.session_state.pop("summary_answer", None)
                    st.success("Video processing successful")

                except Exception as e:
                    st.error(f"Error during transcript processing: {e}")

# 3. Interactive Section with Tabs
if "vectorstore" in st.session_state:
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["Q&A", "Video Summary", "Quiz & Progress"])
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
    except Exception as e:
        llm = None

    video_id = st.session_state["video_id"]
    
    # --- TAB 1: Q&A CHAT (TEXT & VOICE) ---
    with tab1:
        st.subheader("Ask Questions (Via Text or Voice 🎙️)")
        
        input_mode = st.radio("Choose Input Mode:", ["Text Input", "Voice Recording 🎙️"], horizontal=True)
        user_query = None
        
        if input_mode == "Voice Recording 🎙️":
            st.info("Add your question by speaking.")
            audio_value = st.audio_input("Speak your question here")
            if audio_value:
                with st.spinner("Transcribing your voice "):
                    try:
                        audio_bytes = audio_value.getvalue()
                        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
                        
                        stt_message = HumanMessage(content=[
                            {"type": "text", "text": "Please transcribe this audio clip into text accurately. Output only the exact transcribed question text."},
                            {"type": "media", "data": encoded_audio, "mime_type": "audio/wav"}
                        ])
                        trans_res = llm.invoke([stt_message])
                        user_query = trans_res.text if hasattr(trans_res, "text") else str(trans_res.content)
                        st.success(f"Transcribed Question: **{user_query}**")
                    except Exception as e:
                        st.error(f"Voice transcription error: {e}")
        else:
            st.markdown("**Quick Suggestions:**")
            col_s1, col_s2, col_s3 = st.columns(3)
            suggested_query = None
            if col_s1.button("What is the main topic?"):
                suggested_query = "What is the main topic of this video?"
            if col_s2.button("Summarize key takeaways"):
                suggested_query = "What are the key takeaways from this video?"
            if col_s3.button("Explain main concepts"):
                suggested_query = "What are the main concepts or steps explained?"

            user_query_input = st.text_input("Enter your question:", key="qa_query")
            user_query = suggested_query if suggested_query else user_query_input

        if (st.button("Get Answer") or (input_mode == "Voice Recording 🎙️" and user_query)) and user_query:
            if not api_key:
                st.warning("Please enter your API key in the sidebar.")
            else:
                with st.spinner("Generating answer "):
                    try:
                        retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 3})
                        retrieved_docs = retriever.invoke(user_query)
                        
                        context_text = ""
                        for doc in retrieved_docs:
                            t_str = doc.metadata["time_formatted"]
                            t_sec = int(doc.metadata["start_time"])
                            
                            context_text += f"[Timestamp: {t_str} ] {doc.page_content}\n\n"

                        system_prompt = (
                            "You are an AI assistant answering questions based on the provided YouTube video transcript context.\n"
                            "Each context block includes a timestamp link like [Timestamp: MM:SS (URL)].\n"
                            "When answering, include these exact markdown links next to timestamps so the user can click them.\n"
                            "If the answer is not in the context, state that you do not know.\n\n"
                            "Context:\n{context}"
                        )

                        prompt = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{input}"),
                        ])

                        chain = prompt | llm | StrOutputParser()
                        answer = chain.invoke({"context": context_text, "input": user_query})

                        # 1. Write Answer on Screen
                        st.markdown(f"**Query Asked:** *{user_query}*")
                        st.markdown("### Answer:")
                        st.markdown(answer)

                        # 2. Give Answer Through Voice (Text-to-Speech)
                        try:
                            clean_text_for_speech = re.sub(r'\[.*?\]\(.*?\)', '', answer) # Markdown links remove for clean voice
                            tts = gTTS(text=clean_text_for_speech, lang='en')
                            audio_buffer = io.BytesIO()
                            tts.write_to_fp(audio_buffer)
                            audio_buffer.seek(0)
                            st.audio(audio_buffer, format="audio/mp3")
                        except Exception as tts_err:
                            st.warning(f"Voice generation warning: {tts_err}")

                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            st.error("API Quota Exceeded! You have hit your free tier daily request limit.")
                        else:
                            st.error(f"Answer generation error: {e}")

    # --- TAB 2: VIDEO CHAPTERS & SUMMARY ---
    with tab2:
        st.subheader("Summary of the video")
        if st.button("Generate Summary"):
            with st.spinner("Analyzing"):
                try:
                    summary_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are an expert video summarizer. Analyze the text and break it down into chronological chapters or key sections with descriptions."),
                        ("human", "Transcript Summary Text: {transcript}")
                    ])
                    summary_chain = summary_prompt | llm | StrOutputParser()
                    st.session_state["summary_answer"] = summary_chain.invoke({"transcript": st.session_state['full_text'][:15000]})
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        st.error("API Quota Exceeded! Daily free request limit reached.")
                    else:
                        st.error(f"Summary generation error: {e}")

        if "summary_answer" in st.session_state:
            st.markdown(st.session_state["summary_answer"])
            st.download_button(
                label="Download Summary as TXT",
                data=st.session_state["summary_answer"],
                file_name="video_summary.txt",
                mime="text/plain"
            )

    # --- TAB 3: INTERACTIVE QUIZ & PROGRESS TRACKER ---
    with tab3:
        st.subheader("Test Your Understanding ")
        
        if "quiz_data" not in st.session_state:
            if st.button("Generate Interactive Quiz"):
                with st.spinner("Building structured quiz questions..."):
                    try:
                        quiz_system = (
                            "You are an educational quiz generator. Create 3 multiple choice questions based on the transcript text.\n"
                            "IMPORTANT: Regardless of the language of the transcript text provided (whether Hindi or any other language), you MUST generate all questions, options, and answers strictly in English.\n"
                            "You MUST output your response strictly as a valid JSON array of objects with keys: 'question', 'options' (list of 4 strings), and 'answer' (exact matching choice string). Do not include any text outside the JSON array."
                        )
                        
                        quiz_prompt = ChatPromptTemplate.from_messages([
                            ("system", quiz_system),
                            ("human", "Transcript Text: {transcript}")
                        ])
                        
                        quiz_chain = quiz_prompt | llm | StrOutputParser()
                        raw_quiz = quiz_chain.invoke({"transcript": st.session_state['full_text'][:15000]})
                        
                        cleaned_json = re.sub(r"^```json\s*|\s*```$", "", raw_quiz.strip(), flags=re.MULTILINE)
                        cleaned_json = re.sub(r"^```\s*|\s*```$", "", cleaned_json.strip(), flags=re.MULTILINE)
                        
                        st.session_state["quiz_data"] = json.loads(cleaned_json)
                        st.rerun()
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            st.error("API Quota Exceeded! Daily free request limit reached for Gemini.")
                        else:
                            st.error(f"Could not generate interactive quiz. Error: {e}")
        
        if "quiz_data" in st.session_state:
            quiz_list = st.session_state["quiz_data"]
            total_questions = len(quiz_list)
            
            user_answers = {}
            answered_count = 0
            
            st.markdown("### Quiz Progress")
            
            with st.form(key="interactive_quiz_form"):
                for idx, q in enumerate(quiz_list):
                    st.markdown(f"**Q{idx+1}: {q['question']}**")
                    choice = st.radio(
                        f"Select option for Q{idx+1}:", 
                        q['options'], 
                        index=None, 
                        key=f"q_{idx}"
                    )
                    user_answers[idx] = choice
                    if choice is not None:
                        answered_count += 1
                    st.divider()
                
                progress_ratio = answered_count / total_questions if total_questions > 0 else 0
                st.progress(progress_ratio, text=f"Completed {answered_count} out of {total_questions} questions")
                
                submitted = st.form_submit_button("Submit Quiz & Check Score")
                
                if submitted:
                    score = 0
                    result_report = "### Quiz Results:\n"
                    for idx, q in enumerate(quiz_list):
                        user_choice = user_answers.get(idx)
                        correct_choice = q['answer']
                        
                        if user_choice == correct_choice:
                            score += 1
                            result_report += f"- Q{idx+1}: Correct! (Your choice: {user_choice})\n"
                            st.success(f"Q{idx+1}: Correct! (Your choice: {user_choice})")
                        elif user_choice is None:
                            result_report += f"- Q{idx+1}: Not answered. Correct was: {correct_choice}\n"
                            st.warning(f"Q{idx+1}: Not answered. The correct answer was: **{correct_choice}**")
                        else:
                            result_report += f"- Q{idx+1}: Incorrect. (Your choice: {user_choice} | Correct: {correct_choice})\n"
                            st.error(f"Q{idx+1}: Incorrect. (Your choice: {user_choice}) | Correct Answer: **{correct_choice}**")
                    
                    st.metric(label="Final Score", value=f"{score} / {total_questions}")
                    st.session_state["quiz_result_report"] = result_report

            if "quiz_result_report" in st.session_state:
                st.download_button(
                    label="Download Quiz Report",
                    data=st.session_state["quiz_result_report"],
                    file_name="quiz_report.txt",
                    mime="text/plain"
                )
            
            if st.button("Reset Quiz"):
                st.session_state.pop("quiz_data", None)
                st.session_state.pop("quiz_result_report", None)
                st.rerun()