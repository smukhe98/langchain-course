from langchain_core.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

tavily = TavilySearch()

#@tool
#def search(query: str) -> str:
#    """Search the web for information about the query"""
#    print(f"Searching the web for information about {query}")
#    return tavily.search(query=query)


load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str =Field(description="The URL of the Source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer:str = Field(description="the agent's answer to the query")
    sources: List[Source] = Field(default_factory = list, description="List of sources used to generate the answer")

tools =  [tavily]
llm=ChatOllama(model="qwen3:1.7b", temperature=0)
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello, World!")
    response = agent.invoke({"messages": [HumanMessage(content="Search for 2 job postings for an ai engineer using langchain in India on linkedin and list their details")]})
    print(response["messages"][-1].content)


# def main():
#     print("Hello, World!")
#     information = """
#     Elon Reeve Musk (/ˈiːlɒn/ ⓘ; born June 28, 1971) is an industrialist, futurist, and former United States public official. He is the chief executive officer and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire (in terms of US dollars) in June 2026; as of November 2026, Forbes estimates his net worth to be $849.7 billion.

# Musk was born into the wealthy Musk family in Pretoria, South Africa. He emigrated to Canada in 1989; he has Canadian citizenship as his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California, United States, to pursue business ventures. In 1995, he co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal. Musk also became an American citizen in 2002.

# In 2002, Musk founded and became CEO and chief engineer of SpaceX, a space technology company. Musk joined Tesla as an angel investor in 2004 and became its CEO and product architect in 2008. In 2015, Musk co-founded OpenAI to advance artificial intelligence (AI) research, but later left; his growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired Twitter, a social networking service, and made significant changes, including rebranding it as X in 2023. His other businesses include Neuralink, a neurotechnology company that he co-founded in 2016, and the Boring Company, a tunneling company that he founded in 2017. In November 2025, Tesla approved a pay package worth $1 trillion for Musk, which he is to receive over 10 years if certain milestones are met, such as achieving a market capitalization of $8.5 trillion. Musk became the first US-dollar trillionaire upon the initial public offering of SpaceX in June 2026, though a SpaceX stock collapse brought him under the threshold by the next month.


#     """
#     summary_template = """
#     Given the information: {information} about a person, I want you to create:
#     1. A short summary of the person
#     2. Two interesting fun facts about the person
#     """

#     summary_prompt_template = PromptTemplate(
#         input_variables=["information"],
#         template=summary_template
#     )
#     #llm = ChatOpenAI(model="gpt-5", temperature=0)
#     llm=ChatOllama(model="gemma3:270m", temperature=0)
#     chain = summary_prompt_template | llm

#     response = chain.invoke({"information": information})
#     print(response.content)

if __name__ == "__main__":
    main()