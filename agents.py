from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from pydantic import BaseModel
from typing import Literal

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# Router
class RouteDecision(BaseModel):
    category: Literal["RESEARCH", "CASUAL"]


router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query router.

Classify the user's input into exactly ONE category:

RESEARCH
- The user wants detailed, factual, current, or researched information.
- Examples:
  "Research artificial intelligence"
  "What are the latest developments in AI?"
  "Give me a detailed report on climate change"

CASUAL
- Normal conversation, greetings, simple questions, opinions, or general chat.
- Examples:
  "hello"
  "how are you?"
  "what can you do?"
  "thanks"

Respond with exactly one category."""
    ),
    ("human", "{query}")
])

router_chain = router_prompt | llm.with_structured_output(RouteDecision)


# First agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""You are a web research agent.

Your job is to:

1. Understand the user's research topic.
2. Use the web_search tool to find recent and reliable information.
3. Prefer trustworthy and relevant sources.
4. Return useful titles, URLs, and factual snippets.
5. Do not invent sources or information.
6. Do not open, scrape, or visit URLs yourself.
7. Use only the web_search tool.
"""
    )


# Second agent
def build_fetch_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""You are a research reader agent.

Your job is to:
1. Identify the most relevant URL from the search results.
2. Use the scrape_url tool to retrieve its content.
3. Return useful factual information from the scraped content.

Do not perform web searches.
Use only the scrape_url tool.
"""
    )


# Writer
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Write clear, structured and insightful reports."
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:

{research}

Structure the report as:

- Introduction

- Key Findings (minimum 3 well-explained points)

- Conclusion

- Sources (list all URLs found in the research)

Be detailed, factual and professional."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# Critic
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict research report critic.

Evaluate the quality of the given research report based on:

- Accuracy
- Relevance
- Completeness
- Clarity
- Structure
- Quality of reasoning
- Use of sources

Respond ONLY in the following format:

Score: X/10

Strengths:

- ...
- ...
- ...

Areas to Improve:

- ...
- ...
- ...

Verdict: <one concise sentence>

Be objective and critical. Do not rewrite the report."""
    ),
    (
        "human",
        """Evaluate the following research report:

{report}

Respond using exactly the format specified in the system instructions."""
    )
])

critic_chain = critic_prompt | llm | StrOutputParser()