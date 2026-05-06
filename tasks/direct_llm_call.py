from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("human",
     "You are an assistant for question-answering tasks. Use the following retrieved context to answer the question. "
     "If you don't know the answer, say that you don't know. Keep the answer concise.\n\n"
     "Context: {context}\n\nQuestion: {question}\n\nAnswer:")
])

# LLM
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def llm_call(docs, question):
    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": docs, "question": question})
    return generation