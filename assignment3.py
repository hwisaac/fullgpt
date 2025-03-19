from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler


chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

# 예제 데이터
examples = [
    {"input": "Top Gun", "output": "🛩️👨‍✈️🔥"},
    {"input": "The Godfather", "output": "👨‍👨‍👦🔫🍝"},
    {"input": "Finding Nemo", "output": "🐠👨‍👦🌊"},
    {"input": "Titanic", "output": "🚢❤️💔"},
]

# 예제 프롬프트 템플릿 정의
example_template = PromptTemplate(
    input_variables=["input", "output"], template="Movie: {input}\nEmojis: {output}"
)

# FewShotPromptTemplate 설정
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="You are a movie bot that responds with exactly three emojis representing a movie.",
    suffix="Movie: {movie_title}\nEmojis:",
    input_variables=["movie_title"],
)

# 메모리 설정 - ConversationBufferMemory 사용
memory = ConversationBufferMemory(input_key="movie_title", memory_key="chat_history")

# LLMChain을 사용한 영화 이모티콘 변환 체인
movie_emoji_chain = LLMChain(
    llm=chat, prompt=few_shot_prompt, memory=memory, verbose=True
)

# 체인 테스트
if __name__ == "__main__":
    print("\n--- 첫 번째 영화 ---")
    result1 = movie_emoji_chain.run(movie_title="인터스텔라")
    print(f"영화: 인터스텔라\n이모티콘: {result1}")

    print("\n--- 두 번째 영화 ---")
    result2 = movie_emoji_chain.run(movie_title="해리 포터")
    print(f"영화: 해리 포터\n이모티콘: {result2}")

    print("\n--- 메모리 테스트 ---")
    # 이전에 물어본 영화를 확인하기 위한 새로운 체인 생성
    # 메모리의 내용을 불러오기
    chat_history = memory.load_memory_variables({})["chat_history"]

    # 새로운 프롬프트와 체인을 사용
    memory_test_prompt = PromptTemplate(
        template="다음은 이전 대화 기록입니다:\n{chat_history}\n\n위 대화에서 언급된 영화 제목들을 알려주세요.",
        input_variables=["chat_history"],
    )

    # 이 체인에는 메모리를 연결하지 않습니다
    memory_test_chain = LLMChain(llm=chat, prompt=memory_test_prompt, verbose=True)

    memory_result = memory_test_chain.run(chat_history=chat_history)
    print("메모리 테스트 결과:")
    print(memory_result)
