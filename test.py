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





if __name__ == "__main__":
    from tools.tavily_tool import tavily_search
    from tools.weather_tool import get_current_weather

    print(tavily_search("Best hotels in Tokyo"))
    print(get_current_weather("Tokyo"))


