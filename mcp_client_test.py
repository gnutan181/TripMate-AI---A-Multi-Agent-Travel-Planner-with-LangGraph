import os
import asyncio
import certifi
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient


os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


load_dotenv()


TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
# AVIATIONSTACK_API_KEY=os.getenv("AVIATIONSTACK_API_KEY")


client = MultiServerMCPClient(
    {
        "tavily":{
            "transport":"streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        }
    }
)

async def get_all_tools():
    tools = await client.get_tools()
    print("\nAvailable MCP Tools:")


    for tool in tools:
        print(tool.name)


        
tavily_search_tool = None
async def get_tavily_search_tool():
    global tavily_search_tool

    # Already initialized
    if tavily_search_tool is not None:
        return tavily_search_tool

    # Get all tools from MCP
    tools = await client.get_tools()

    print("\nAvailable MCP Tools:")

    for tool in tools:
        print(tool.name)

    # Find Tavily tool
    tavily_search_tool = next(
        (tool for tool in tools if tool.name == "tavily_search"),
        None
    )

    if tavily_search_tool is None:
        raise RuntimeError("Tavily search tool not found")

    return tavily_search_tool


async def tavily_mcp_search(query: str):
    tool = await get_tavily_search_tool()

    result = await tool.ainvoke(
        {
            "query": query
        }
    )

    return result