# Cloudflare 공식문서를 위한 SiteGPT 버전을 만드세요.
# 챗봇은 아래 프로덕트의 문서에 대한 질문에 답변할 수 있어야 합니다:
# [AI Gateway](https://developers.cloudflare.com/ai-gateway/)
# [Cloudflare Vectorize](https://developers.cloudflare.com/vectorize/)
# [Workers AI](https://developers.cloudflare.com/workers-ai/)
# [사이트맵](https://developers.cloudflare.com/sitemap-0.xml을 사용하여 각 제품에 대한 공식문서를 찾아보세요.
# 여러분이 제출한 내용은 다음 질문으로 테스트됩니다:
# - "llama-2-7b-chat-fp16 모델의 1M 입력 토큰당 가격은 얼마인가요?"
# - "Cloudflare의 AI 게이트웨이로 무엇을 할 수 있나요?"
# - "벡터라이즈에서 단일 계정은 몇 개의 인덱스를 가질 수 있나요?"
# 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
# st.sidebar를 사용하여 Streamlit app과 함께 깃허브 리포지토리에 링크를 넣습니다.

from langchain.document_loaders import WebBaseLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

st.set_page_config(
    page_title="Cloudflare SiteGPT",
    page_icon="🖥️",
)

st.markdown(
    """
    # Cloudflare SiteGPT
            
    Cloudflare의 AI 관련 제품들에 대해 질문해보세요.
    - AI Gateway
    - Cloudflare Vectorize
    - Workers AI
"""
)

with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    st.markdown("[GitHub Repo](https://github.com/hwisaac/fullgpt)")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=openai_api_key,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\n" for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


@st.cache_data(show_spinner="Loading Cloudflare documentation...")
def load_website():
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    # 세 가지 제품의 문서 URL만 사용
    urls = [
        "https://developers.cloudflare.com/ai-gateway/",
        "https://developers.cloudflare.com/vectorize/",
        "https://developers.cloudflare.com/workers-ai/",
        "https://developers.cloudflare.com/workers-ai/platform/pricing/",
    ]

    loader = WebBaseLoader(urls)
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(api_key=openai_api_key))
    return vector_store.as_retriever()


retriever = load_website()
query = st.text_input("Cloudflare AI 제품들에 대해 질문해보세요.")
if query:
    chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )
    result = chain.invoke(query)
    st.markdown(result.content.replace("$", "\$"))
