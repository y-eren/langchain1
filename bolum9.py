from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

model = ChatOpenAI(model="gpt-4")
memory = SqliteSaver.from_conn_string(":memory:")
search = TavilySearchResults(max_results=2)

tools = [search]

# model_with_tools = model.bind_tools(tools) # modelleri toollara bağlayabiliyoruz

agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable" : {"thread_id" : "abc123"}}




if __name__ == "__main__":
    # search_result = search.invoke("what is the weather in istanbul? ")
    # response = model_with_tools.invoke([HumanMessage(content="what is the weather in istanbul?")])
    # print(response)

    # response = agent_executor.invoke({"messages" : [HumanMessage(content="What is the weather in istnabul now")]})
    #
    # print(response.content)
    while True:
        user_input = input(">")
        for chunk in agent_executor.stream({"messages" : [HumanMessage(content=user_input)]},
                                           config=config):
            print(chunk)

