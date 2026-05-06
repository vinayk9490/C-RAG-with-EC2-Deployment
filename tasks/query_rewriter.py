### Question Re-writer

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)


def rewrite_prompt(user_question):
    # Prompt
    system = """You are a question re-writer that converts an input question to a better version optimized for web search.
        Return only the improved question, keep it concise and under 100 words."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {user_question} \n Formulate an improved question.",
            ),
        ]
    )
    question_rewriter = re_write_prompt | llm
    response = question_rewriter.invoke({"user_question": user_question})
    return response.content