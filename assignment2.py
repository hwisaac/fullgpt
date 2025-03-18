from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

# 예제 데이터
examples = [
    {
        "movie": "Inception",
        "answer": """
        Title: Inception
        Director: Christopher Nolan
        Cast: Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page
        Budget: $160 million
        Box Office: $829.9 million
        Genre: Science Fiction, Thriller
        Synopsis: A thief who enters the dreams of others to steal secrets is given a chance to have his criminal history erased if he can successfully plant an idea into a target's subconscious.
        """,
    },
    {
        "movie": "Titanic",
        "answer": """
        Title: Titanic
        Director: James Cameron
        Cast: Leonardo DiCaprio, Kate Winslet, Billy Zane
        Budget: $200 million
        Box Office: $2.195 billion
        Genre: Romance, Drama
        Synopsis: A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.
        """,
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Give me information about the movie '{movie}'"),
        ("ai", "{answer}"),
    ]
)

example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a movie expert. Provide structured information about movies in the given format.",
        ),
        example_prompt,
        ("human", "Give me information about the movie '{movie}'"),
    ]
)

chain = final_prompt | chat

# 테스트 실행
chain.invoke({"movie": "The Dark Knight"})
