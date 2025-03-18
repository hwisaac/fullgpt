from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

# OpenAI 모델 설정
chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

# 프로그래밍 언어에 대한 시를 생성하는 체인
poetry_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a poet specializing in writing poetry about programming languages.",
        ),
        ("human", "Write a short poem about the {language} programming language."),
    ]
)

poetry_chain = poetry_prompt | chat

# 시를 설명하는 체인
explanation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in literary analysis. You explain poetry in an insightful way.",
        ),
        ("human", "Explain the meaning and themes of the following poem:\n\n{poem}"),
    ]
)

explanation_chain = explanation_prompt | chat

# LCEL을 사용하여 두 체인 연결
final_chain = {"poem": poetry_chain} | explanation_chain


# 실행
def generate_poetry_and_explanation(language: str):
    response = final_chain.invoke({"language": language})
    return response


# 테스트 실행
print(generate_poetry_and_explanation("Python"))
