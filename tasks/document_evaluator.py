from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


from langchain_groq import ChatGroq
from dotenv import load_dotenv
print(load_dotenv())

import os
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
def documents_evaluator(document, question):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader
    result = retrieval_grader.invoke({"document": document, "question": question})
    return result.binary_score