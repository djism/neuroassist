from pydantic import BaseModel, Field
from typing import Optional


class QuestionRequest(BaseModel):
    """
    Request body for the /ask endpoint.
    """
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The question to ask NeuroAssist",
        examples=["What does the Shrestha lab study?"]
    )


class AnswerResponse(BaseModel):
    """
    Response body from the /ask endpoint.
    """
    answer: str = Field(
        description="The generated answer from the agent"
    )
    sources: list[str] = Field(
        default=[],
        description="Source files or collections used to generate the answer"
    )
    route: str = Field(
        description="Which retriever was used: papers, code, graph, pubmed, papers_and_code"
    )
    faithfulness_score: Optional[float] = Field(
        default=None,
        description="RAGAS faithfulness score (0.0-1.0). None if evaluation was skipped."
    )
    eval_error: Optional[str] = Field(
        default=None,
        description="Evaluation error message if RAGAS scoring failed"
    )


class HealthResponse(BaseModel):
    """
    Response body for the /health endpoint.
    """
    status: str
    collections: dict
    message: str


class CollectionStats(BaseModel):
    """
    Stats about a ChromaDB collection.
    """
    count: int
    status: str


if __name__ == "__main__":
    print("Testing schemas...\n")

    # Test request
    req = QuestionRequest(question="What does the Shrestha lab study?")
    print(f"Request  : {req.question}")

    # Test response
    resp = AnswerResponse(
        answer="The Shrestha lab studies...",
        sources=["Lab_Paper1.pdf"],
        route="graph",
        faithfulness_score=0.92,
        eval_error=None
    )
    print(f"Answer   : {resp.answer}")
    print(f"Route    : {resp.route}")
    print(f"Score    : {resp.faithfulness_score}")

    print("\n✅ Schemas working correctly!")