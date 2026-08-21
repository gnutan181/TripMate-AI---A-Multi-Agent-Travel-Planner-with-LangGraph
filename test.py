# from tools.tavily_tool import tavily_search
# from tools.flight_tool import search_flights
# from backend import run_travel_agent

# ||||||||||||||||||||||||||||| tavliy api ||||||||||||||||||||
# res = tavily_search("Best hotels in india")
# print(res)


# ||||||||||||||||||||||||||||| Aviation api  ||||||||||||||||||||
# res = search_flights("Plan a complete 7 days India  trip from Dubai including flights, hotels, and sightseeing under 2 lakhs.")
# print(res)

# user_input = input("Enter travel request:")
# response = run_travel_agent(
#     user_input= user_input,
#     thread_id="test_user"
# )
# print("\nFINAL RESPONSE:\n")
# print(response["answer"])





# ||||||||||||||||||||||||||||| tavliy mcp test ||||||||||||||||||||

# import asyncio

# from mcp_client_test import get_all_tools, tavily_mcp_search

# if __name__ == "__main__":

#     query = "latest news about AI"
#     # asyncio.run(get_all_tools())
#     res = asyncio.run(tavily_mcp_search(query))
#     print("res",res)



# ||||||||||||||||||||||||||||| Aviation mcp test ||||||||||||||||||||

import asyncio

from mcp_client import get_all_tools

if __name__ == "__main__":
    asyncio.run(get_all_tools())
    print()


