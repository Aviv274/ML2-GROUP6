import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from typing import Annotated, TypedDict
import datetime
import operator
from io import BytesIO
from fpdf import FPDF
import re
from hotel_tool import hotels_finder, HotelsInput
from flight_tool import flights_finder, FlightsInput
import json


# Load environment variables
load_dotenv()

# Constants
CURRENT_YEAR = datetime.datetime.now().year

# App config
st.set_page_config(page_title="AI Travel Agent", page_icon="🌍", layout="wide")
st.title("🌍 AI Travel Agent")

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# System prompt with DETAILED example tool call
TOOLS_SYSTEM_PROMPT = f"""You are a smart travel agency. Use the tools to look up information.
You are allowed to make multiple calls (either together or in sequence).
Only look up information when you are sure of what you want.
The current year is {CURRENT_YEAR}.

In your output always include:
- name and rating of the hotel
- price per night and total cost (with currency symbol, e.g., €84 per night, €337 total)
- and a link if possible
- for flights: airline name, price, departure and arrival airports, departure and arrival times, and a booking link

Before the day-by-day itinerary, always show a short summary of the chosen hotel and chosen flight(s), including all relevant information.

Return your results in this markdown table format, per day:
**Day X: Title**
| Time | Activity | Status |
|------|----------|--------|
| 09:00 | Visit the Louvre Museum | Not Yet Started |
| 14:00 | Lunch at Café de Flore | Not Yet Started |

Use 24-hour format (e.g., 14:00) for all times.
Do not use "General" in the itinerary table. Always include a real time, even for flights and hotel check-ins. If the time is flexible or unknown, make a reasonable estimate (e.g., 14:00 for check-in).
Only include "General" as time row for hotel or flight or similar overall items.

Adjust the travel itinerary to start only after the flight's landing time, and begin near the arrival airport.

Always plan to arrive at the airport **at least 3 hours before any international flight departure**. Adjust the itinerary accordingly to allow for enough travel and check-in time.

Always generate a rich, full itinerary that includes specific recommended places to visit, eat and enjoy.
Use famous, popular, or hidden gem recommendations in the area of the hotel or arrival airport.

Do not be vague. For example, instead of:
"Visit a Parisian landmark" → say "Visit the Eiffel Tower"
"Explore nightlife" → say "Have a drink at Little Red Door, one of Paris’ top speakeasies"

Your output must feel like a personal guide built by a local expert, not a generic outline.
If you need to look up some information before asking a follow up question, you are allowed to do that!
I want to have in your output links to hotels or flights websites (if possible).

Use the following logic for hotel_class selection based on budget:
- Low budget: hotel_class = "1,2"
- Medium budget: hotel_class = "3,4"
- High budget: hotel_class = "5"

Example Tool Calls:

hotels_finder({{
    "q": "Paris",
    "check_in_date": "2024-07-01",
    "check_out_date": "2024-07-05",
    "adults": 2,
    "children": 1,
    "rooms": 1,
    "hotel_class": "3,4",
    "sort_by": 8
}})

flights_finder({{
    "departure_airport": "JFK",
    "arrival_airport": "CDG",
    "outbound_date": "2024-07-01",
    "return_date": "2024-07-05",
    "adults": 2,
    "children": 0
}})

When booking flights, automatically determine the closest major airport to a given city using a predefined mapping (e.g., Paris → CDG, Madrid → MAD). Do not ask the user for airport codes. If the city is not in the mapping, make a reasonable assumption based on well-known airport locations.

Please include complete flight information — both outbound (origin to destination) and return (destination to origin) segments.
"""


# Define agent tools
TOOLS = [hotels_finder,flights_finder]

# Build the agent class
class Agent:
    def __init__(self):
        self._tools = {t.name: t for t in TOOLS}
        self._tools_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        ).bind_tools(TOOLS)

        builder = StateGraph(AgentState)
        builder.add_node("call_tools_llm", self.call_tools_llm)
        builder.add_node("invoke_tools", self.invoke_tools)
        builder.set_entry_point("call_tools_llm")

        builder.add_conditional_edges("call_tools_llm", self.exists_action, {
            "more_tools": "invoke_tools",
            "end": END
        })
        builder.add_edge("invoke_tools", "call_tools_llm")

        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory)

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        if hasattr(result, "tool_calls") and len(result.tool_calls) > 0:
            return "more_tools"
        return "end"

    def call_tools_llm(self, state: AgentState):
        messages = [SystemMessage(content=TOOLS_SYSTEM_PROMPT)] + state["messages"]
        message = self._tools_llm.invoke(messages)
        return {"messages": [message]}

    def invoke_tools(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            if t["name"] not in self._tools:
                result = "Invalid tool"
            else:
                args = t.get("args", {})

                try:
                    if t["name"] == "hotels_finder":
                        if "q" not in args:
                            args["q"] = st.session_state.get("destination", "")
                        if "check_in_date" not in args:
                            args["check_in_date"] = str(st.session_state.get("start_date", datetime.date.today()))
                        if "check_out_date" not in args:
                            args["check_out_date"] = str(st.session_state.get("end_date", datetime.date.today()))
                        if "adults" not in args:
                            args["adults"] = 2
                        if "hotel_class" not in args:
                            budget = st.session_state.get("budget", "Medium").lower()
                            if budget == "low":
                                args["hotel_class"] = "1,2"
                            elif budget == "medium":
                                args["hotel_class"] = "3,4"
                            elif budget == "high":
                                args["hotel_class"] = "5"
                        if "sort_by" not in args:
                            budget = st.session_state.get("budget", "Medium").lower()
                            args["sort_by"] = "3" if budget == "low" else "8"

                        parsed_args = HotelsInput(**args)

                    elif t["name"] == "flights_finder":
                        with open("cities_iata.json", "r", encoding="utf-8") as f:
                            cities_iata = json.load(f)
                        
                        if "departure_airport" not in args:
                            args["departure_airport"] = cities_iata.get(st.session_state.get("origin", "").lower())
                        if "arrival_airport" not in args:
                            args["arrival_airport"] = cities_iata.get(st.session_state.get("destination", "").lower())
                        if "outbound_date" not in args:
                            args["outbound_date"] = str(st.session_state.get("start_date", datetime.date.today()))
                        if "return_date" not in args:
                            args["return_date"] = str(st.session_state.get("end_date", datetime.date.today()))
                        if "adults" not in args:
                            args["adults"] = st.session_state.get("adult", 1)
                        if "children" not in args:
                            args["children"] = st.session_state.get("children", 0)


                        parsed_args = FlightsInput(**args)

                    else:
                        result = "Unsupported tool"
                        results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
                        continue

                    result = self._tools[t["name"].strip()].invoke({"params": parsed_args})

                except Exception as e:
                    result = f"Tool call failed: {e}"

            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
        return {"messages": results}

# Instantiate the agent
agent = Agent()

# Sidebar preferences form
with st.sidebar:
    st.header("📂 Travel Preferences")
    origin = st.text_input("From where would you like to travel?")
    destination = st.text_input("Where would you like to travel?")
    start_date = st.date_input("Start date", datetime.date.today())
    end_date = st.date_input("End date", datetime.date.today())
    budget = st.selectbox("What is your budget level?", ["Low", "Medium", "High"])
    interests = st.text_area("Main interests (e.g., food, museums, nature, nightlife)")
    avoid = st.text_input("Anything to avoid?")
    adult = st.number_input("Number of adult", min_value=0, value=1)
    children = st.number_input("Number of children", min_value=0, value=0)
    if st.button("🧽 Generate Trip Plan"):
        user_message = f"""
Create a personalized itinerary.
origin: {origin}
Destination: {destination}
Start Date: {start_date}
End Date: {end_date}
Budget: {budget}
Interests: {interests}
Avoid: {avoid}
children: {children}
adult: {adult}
"""
        st.session_state.user_prompt = user_message
        st.session_state.origin = origin
        st.session_state.destination = destination
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.adult = adult
        st.session_state.chat_history = [HumanMessage(content=user_message)]
        st.rerun()

# Run agent if user_prompt exists
if "user_prompt" in st.session_state:
    with st.spinner("Planning your trip..."):
        valid_messages = [msg for msg in st.session_state.chat_history if getattr(msg, "content", "").strip()]
        if valid_messages:
            events = agent.graph.invoke(
                {"messages": valid_messages},
                config={"thread_id": "travel_agent_session"}
            )
            response = events["messages"][-1]
            st.session_state.chat_history.append(response)
        else:
            st.warning("Please enter a valid message before generating the trip plan.")

    st.markdown("### 💬 Chat with Your AI Travel Agent")
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            bubble_style = "background-color:rgba(224,247,250,0.7); color:#000;"
        else:
            bubble_style = "background-color:rgba(240,240,240,0.7); color:#000;"
        st.markdown(f"<div style='{bubble_style} padding:10px; border-radius:10px; margin:5px;'>{msg.content}</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Ask to adjust your trip:")
    if user_input and user_input.strip():
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.rerun()
