from langchain_tavily import TavilySearch

def tavily_results():
    web_search_tool = TavilySearch(max_results=3)
    return web_search_tool