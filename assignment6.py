import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
import os
from langchain.schema.runnable import RunnableLambda, RunnableMap


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

# --- Sidebar ---
with st.sidebar:
    st.subheader("Settings")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    difficulty = st.selectbox("Select difficulty", ["Easy", "Medium", "Hard"])
    st.markdown("[GitHub Repo](https://github.com/hwisaac/fullgpt)")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.1,
    # 실제 모델명으로 바꾸세요 (예: gpt-3.5-turbo 등)
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.
            Based ONLY on the following context, make 10 (TEN) {difficulty} questions to test the user's knowledge.
            Each question should have 4 answers, three incorrect and one correct, marked with (o).
            Context: {context}
            """,
        )
    ]
)

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You format exam questions into JSON.
            Questions: {context}
            """,
        )
    ]
)

questions_chain = (
    RunnableMap(
        {
            "context": lambda x: format_docs(x["docs"]),
            "difficulty": lambda x: x["difficulty"],
        }
    )
    | questions_prompt
    | llm
)
formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(_term):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(_term)


@st.cache_data(show_spinner="Generating quiz...")
def run_quiz_chain(_docs, difficulty):
    raw = questions_chain.invoke({"docs": _docs, "difficulty": difficulty})
    formatted = formatting_chain.invoke({"context": raw.content})
    return output_parser.parse(formatted.content)


# --- Main Code ---

# docs를 매번 None으로 초기화하고, 선택한 방식을 통해서만 문서를 할당
docs = None
topic = None

choice = st.sidebar.selectbox("Choose input source", ("File", "Wikipedia Article"))

if choice == "File":
    file = st.file_uploader(
        "Upload a .docx, .txt or .pdf file",
        type=["pdf", "txt", "docx"],
        key="file_upload",
    )
    if file:
        docs = split_file(file)
else:  # "Wikipedia Article"
    topic = st.text_input("Search Wikipedia...", key="wiki_topic")
    if topic:
        docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT!
    Upload a file or search a Wikipedia topic to get started.
    """
    )
    st.stop()

# --- Quiz Logic ---
if "score" not in st.session_state:
    st.session_state.score = 0
if "attempted" not in st.session_state:
    st.session_state.attempted = False

quiz_data = run_quiz_chain(docs, difficulty)

with st.form("quiz_form"):
    st.write("### Quiz")
    answers_correct = 0
    user_answers = {}

    for idx, question in enumerate(quiz_data["questions"]):
        st.write(f"**Q{idx+1}: {question['question']}**")

        options_dict = question["options"]
        correct_key = question["correct_answer"]
        correct_text = options_dict[correct_key]

        option_texts = list(options_dict.values())

        selected = st.radio(
            "Select an option", option_texts, index=None, key=f"q_{idx}"
        )

        user_answers[idx] = {
            "selected": selected,
            "correct_text": correct_text,
        }

    submitted = st.form_submit_button("Submit Quiz")

if submitted:
    correct_count = sum(
        1 for ans in user_answers.values() if ans["selected"] == ans["correct_text"]
    )
    total_questions = len(user_answers)

    st.write(f"## Your score: {correct_count} / {total_questions}")

    if correct_count == total_questions:
        st.balloons()
        st.success("Perfect score! Well done!")
    else:
        if st.button("Try Again"):
            st.experimental_rerun()
