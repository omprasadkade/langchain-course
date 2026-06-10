from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()


class Source(BaseModel):
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    answer: str = Field(description="The agent's answer")
    sources: List[Source] = Field(default_factory=list)


llm = ChatOllama(
    model="qwen3",
    temperature=0.9
)

tools = [TavilySearch()]

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse
)


def main():
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is LangChain? Search the web and provide sources."
                )
            ]
        }
    )

    print(result)

    response = result["structured_response"]

    print("\nAnswer:")
    print(response.answer)

    print("\nSources:")
    for source in response.sources:
        print(source.url)

    print("\nAs Dict:")
    print(response.model_dump())


if __name__ == "__main__":
    main()