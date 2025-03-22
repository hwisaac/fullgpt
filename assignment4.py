# Stuff Documents 체인을 사용하여 완전한 RAG 파이프라인을 구현하세요.
# 체인을 수동으로 구현해야 합니다.
# 체인에 ConversationBufferMemory를 부여합니다.
# document.txt 문서를 사용하여 RAG를 수행하세요
# 체인에 다음 질문을 합니다:
# - Aaronson 은 유죄인가요?
# - 그가 테이블에 어떤 메시지를 썼나요?
# - Julia 는 누구인가요?

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.memory import ConversationBufferMemory

# 1. 문서 로드 및 분할
print("1. 문서 로드 및 분할 중...")
loader = UnstructuredFileLoader("./document.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# 2. 벡터 스토어 생성
print("2. 벡터 스토어 생성 중...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()

# 3. LLM 설정
print("3. LLM 설정 중...")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. 메모리 설정
print("4. 메모리 설정 중...")
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

# 5. Stuff Documents 체인 구성
print("5. 체인 구성 중...")

# 프롬프트 템플릿 설정
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

# LLMChain 생성
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question", "chat_history"]
)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# StuffDocumentsChain 생성
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
)

# 6. 수동으로 RAG 체인 구현
print("6. RAG 체인 수행 중...")


# 질문 및 답변 함수 정의
def ask_question(question):
    # 문맥 정보 검색
    docs = retriever.get_relevant_documents(question)

    # 메모리에서 채팅 기록 가져오기
    chat_history = memory.load_memory_variables({})["chat_history"]

    # 체인 실행
    response = stuff_chain.run(
        input_documents=docs, question=question, chat_history=chat_history
    )

    # 메모리 업데이트
    memory.save_context({"question": question}, {"output": response})

    return response


# 첫 번째 질문
print("\n--- 첫 번째 질문: Aaronson 은 유죄인가요? ---")
answer1 = ask_question("Aaronson 은 유죄인가요?")
print(f"답변: {answer1}")

# 두 번째 질문
print("\n--- 두 번째 질문: 그가 테이블에 어떤 메시지를 썼나요? ---")
answer2 = ask_question("그가 테이블에 어떤 메시지를 썼나요?")
print(f"답변: {answer2}")

# 세 번째 질문
print("\n--- 세 번째 질문: Julia 는 누구인가요? ---")
answer3 = ask_question("Julia 는 누구인가요?")
print(f"답변: {answer3}")

# 메모리에 저장된 대화 확인
print("\n--- 메모리에 저장된 대화 내용 ---")
chat_history = memory.load_memory_variables({})["chat_history"]
print(chat_history)
