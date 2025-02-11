import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header") # otomatik bs4 alması için kullanılıyor
        )
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # bu rakamlar değişebilir
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

# rag prompt
prompt = hub.pull("rlm/rag-prompt") # rag için başkasının yazdığı rag prompt kullanılıyor langsmith içerisinde promptlar gözükebiliyor

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = {"context" : retriever | format_docs, "question" : RunnablePassthrough()} | prompt | llm | StrOutputParser()


if __name__ == "__main__":

    for chunk in chain.stream("what is maximum inner product search?"):
        print(chunk, end="", flush=True)

