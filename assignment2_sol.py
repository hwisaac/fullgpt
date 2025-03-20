from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

examples = [
    {
        "movie": "Iron Man",
        "answer": """
        Information of Iron Man
        Director: Jon Favreau
        Main Cast: Robert Downey Jr., Gwyneth Paltrow, Jeff Bridges, Terrence Howard
        Budget: $140 million
        Box Office Revenue: $585.8 million
        Genre: Action, Science Fiction, Superhero
        Synopsis: After being held captive in an Afghan cave, billionaire engineer Tony Stark creates a unique weaponized suit of armor to fight evil.
        """,
    },
    {
        "movie": "Thor",
        "answer": """
        Information of Thor
        Director: Kenneth Branagh
        Main Cast: Chris Hemsworth, Natalie Portman, Tom Hiddleston, Anthony Hopkins
        Budget: $150 million
        Box Office Revenue: $449.3 million
        Genre: Action, Fantasy, Superhero
        Synopsis: The powerful but arrogant god Thor is cast out of Asgard to live amongst humans in Midgard (Earth), where he soon becomes one of their finest defenders.
        """,
    },
    {
        "movie": "Spider-Man",
        "answer": """
        Information of Spider-Man
        Director: Sam Raimi
        Main Cast: Tobey Maguire, Kirsten Dunst, Willem Dafoe, James Franco
        Budget: $139 million
        Box Office Revenue: $825 million
        Genre: Action, Adventure, Superhero
        Synopsis: When bitten by a genetically modified spider, a nerdy high school student gains spider-like abilities that he must use to fight evil as a superhero after tragedy befalls his family.
        """,
    },
]

# FewShotPromptTemplate를 활용

example_prompt = PromptTemplate.from_template(
    "Human: Tell me about {movie} movie.\nAI: {answer}"
)

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix="Human: Tell me about {movie} movie",
    input_variables=["movie"],
)

chain = prompt | chat

chain.invoke({"movie": "The Avengers"})

# FewShotChatMessagePromptTemplate를 활용

example_prompt = ChatPromptTemplate.from_messages(
    [("human", "Tell me about {movie} movie."), ("ai", "{answer}")]
)

fewshot_chat_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a movie expert. Provide detailed and accurate information about movies when asked.",
        ),
        fewshot_chat_prompt,
        ("human", "tell me about {movie} movie."),
    ]
)

chain = final_prompt | chat

chain.invoke({"movie": "The Avengers"})
