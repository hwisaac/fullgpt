import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.memory import ConversationBufferMemory
import openai
import tempfile

# 추가: AIMessage, HumanMessage 임포트
from langchain.schema import AIMessage, HumanMessage

# 스트림릿 페이지 설정
st.set_page_config(page_title="RAG with Streamlit", page_icon=":books:")

# 사이드바: OpenAI API 키 입력
st.sidebar.title("설정")
openai_api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요", type="password")
if openai_api_key:
    openai.api_key = openai_api_key

# 사이드바: 깃허브 링크
st.sidebar.markdown("### Github Repo")
st.sidebar.markdown("[프로젝트 깃허브 링크](https://github.com/your-repo-url)")

st.title("RAG 파이프라인 데모 - 1984 소설 예시")

# 세션 스테이트 초기화
if "memory" not in st.session_state:
    # return_messages=True로 설정하면 HumanMessage, AIMessage 형태로 관리
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

# 파일 업로드
uploaded_file = st.file_uploader(
    "문서 파일을 업로드하세요 (예: 1984 일부 텍스트)", type=["txt"]
)
if uploaded_file is not None:
    st.session_state["uploaded_file"] = uploaded_file


# 벡터 스토어 생성
def create_vectorstore(file):
    # 임시 파일로 저장 (확장자 .txt 명시)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    loader = UnstructuredFileLoader(tmp_file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


# RAG 체인 빌드 함수
def build_rag_chain():
    prompt_template = """
당신은 George Orwell의 1984 소설에 관한 질문에 답변하는 도우미입니다.
주어진 문맥 정보를 기반으로만 답변하고, 문맥 정보에서 발견할 수 없는 내용은 "문맥 정보에서 찾을 수 없습니다"라고 대답하세요.

문맥 정보:
{context}

이전 대화:
{chat_history}

질문: {question}
답변:
"""
    document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"],
    )

    # 필요에 따라 모델명 변경
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
    )
    return stuff_chain


def ask_question(question):
    if st.session_state["vectorstore"] is None:
        return "먼저 문서를 업로드해 주세요."

    retriever = st.session_state["vectorstore"].as_retriever()
    docs = retriever.get_relevant_documents(question)

    # 메모리에서 채팅 기록
    chat_history = st.session_state["memory"].load_memory_variables({})["chat_history"]

    # 체인 실행
    stuff_chain = build_rag_chain()
    response = stuff_chain.run(
        input_documents=docs, question=question, chat_history=chat_history
    )

    # 메모리에 질의/응답 저장
    # return_messages=True 상태에서는
    # "HumanMessage(question)", "AIMessage(response)" 형태로 쌓임
    st.session_state["memory"].save_context(
        {"input": question}, {"output": response}  # "input" 대신 "question" 쓰셔도 무방
    )

    return response


# 파일 업로드 후 벡터 스토어 생성
if st.session_state["uploaded_file"] is not None and openai_api_key:
    if st.session_state["vectorstore"] is None:
        st.write("문서를 벡터 스토어로 변환 중입니다...")
        st.session_state["vectorstore"] = create_vectorstore(
            st.session_state["uploaded_file"]
        )
        st.success("벡터 스토어 생성 완료!")

# 질문 입력
question = st.text_input("질문을 입력하세요:", "")
if st.button("질문하기"):
    if not openai_api_key:
        st.warning("OpenAI API Key를 입력해야 합니다.")
    elif st.session_state["uploaded_file"] is None:
        st.warning("문서 파일을 먼저 업로드해주세요.")
    else:
        with st.spinner("답변 생성 중..."):
            answer = ask_question(question)
        st.write("**Q:**", question)
        st.write("**A:**", answer)

# 이전 대화 히스토리 출력
messages = st.session_state["memory"].chat_memory.messages
if messages:
    st.markdown("---")
    st.markdown("### 대화 기록")
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "사용자"
            content = msg.content
        elif isinstance(msg, AIMessage):
            role = "AI"
            content = msg.content
        else:
            role = "기타"
            content = str(msg)
        st.write(f"**{role}**: {content}")
