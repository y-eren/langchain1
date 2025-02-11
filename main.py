from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory # chat history
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

store = {}

def get_session_history(session_id : str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

model = ChatOpenAI(model = "gpt-3.5-turbo")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability"),
    MessagesPlaceholder(variable_name="messages")
])

chain = prompt_template | model

config = {"configurable" : {"session_id" : "abcde123"}}

# ilk başka model.invoke ile çağırıyorduk daha sonra chain.invoke ile çağrılmaya başlandı

with_message_history = RunnableWithMessageHistory(chain, get_session_history)

if __name__ == '__main__':
    while True:
        user_input = input(">")
        # response = with_message_history.invoke(
        #     [
        #         HumanMessage(content=user_input)
        #     ],
        #     config=config
        # )
        #
        # print(response.content)

        for r in with_message_history.stream(
            [
                HumanMessage(content=user_input),
            ],
            config=config
        ):
            print(r.content, end=" ")