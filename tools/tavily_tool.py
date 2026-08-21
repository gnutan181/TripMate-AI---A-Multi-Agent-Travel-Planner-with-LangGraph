# ||||||||||| This is code is not used because we have used the mcp server for tavily |||||||


# from tavily import TavilyClient
# import os 
# from dotenv import load_dotenv
# load_dotenv()

# client = TavilyClient(
#     api_key=os.getenv("TAVILY_API_KEY")
# )

# # response  = client()


# def tavily_search(query):
#     response = client.search(query=query,max_results=5)
#     result = []

#     for i, r in enumerate(response["results"],1):
#         title = r.get("title","Unknown")
#         url = r.get("url","")
#         snippet = r.get("content","").strip()

#         if len(snippet) > 300 :
#             snippet =snippet[:300].rsplit(" ",1)[0] + "..."

#         result.append(f"{1}. **{title}**\n {url}\n {snippet}")

#     return "\n\n".join(result)

