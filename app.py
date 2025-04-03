import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import datetime
from agent import Agent
import json
import re


# Load environment variables
load_dotenv()

# Constants
CURRENT_YEAR = datetime.datetime.now().year

# App config
st.set_page_config(page_title="AI Travel Assistant", layout="wide")
st.title("🌍 AI Travel Agent")
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
        .card {
            border-radius: 20px;
            padding: 1.5rem;
            background-color: #e3f2f9;
            margin-bottom: 1rem;
        }
        .day-bubble {
            border-radius: 25px;
            padding: 1.5rem;
            background-color: #f0fbff;
            margin-bottom: 2rem;
            border: 1px solid #cce5f6;
        }
        .day-header {
            background-color: #eaf7ff;
            padding: 0.5rem 1rem;
            border-radius: 15px;
            font-size: 1.2rem;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 1rem;
        }
        .activity {
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


# Instantiate the agent
agent = Agent()
response =""

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
            
            ai_msg = events['messages'][1]  # AIMessage object
            content = ai_msg.content        # This is the string that contains the ```json ... ``` block

            # Step 2: Extract JSON from content
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)


            if json_match:
                json_str = json_match.group(1)
                response = json.loads(json_str)
                print("✅ Parsed JSON:")
                print(response)
            else:
                print("❌ JSON block not found in cleaned_str.")
            
        else:
            st.warning("Please enter a valid message before generating the trip plan.")

if response:
    st.markdown(f"### {response['general']}")

    hcol, fcol = st.columns(2)
    with hcol:
        st.markdown(f"""
        <div class='card'>
        <h4>🏨 Hotel</h4>
        <p><strong>{response['hotel']['name']}</strong><br>
        {response['hotel']['price_per_night']} per night<br>
        ⭐ {response['hotel']['rating']}</p>
        <a href="{response['hotel']['link']}" target="_blank">Hotel Website</a>
        </div>
        """, unsafe_allow_html=True)

    with fcol:
        st.markdown(f"""
        <div class='card'>
        <h4>✈️ Flight</h4>
        <p><strong>{response['flight']['airline']}</strong><br>
        {response['flight']['departure_time']} → {response['flight']['arrival_time']}<br>
        {response['flight']['departure_airport']} → {response['flight']['arrival_airport']}<br>
        💰 {response['flight']['price']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🗓️ Itinerary")
    day_map = {
        "visit": "🏛️",
        "explore": "🧭",
        "lunch": "🍽️",
        "dinner": "🍷",
        "check-in": "🏨",
        "check-out": "🧳",
        "arrival": "🛬",
        "departure": "🛫",
        "transfer": "🚗",
        "nightlife": "🍸",
        "shopping": "🛍️",
        "breakfast": "🥐"
    }

    plan_list = response.get("plan", [])
    if isinstance(plan_list, list):
        for day_dict in plan_list:
            for raw_day, activities in day_dict.items():
                day_label = raw_day.replace("day", "Day ").capitalize()
                html = f"<div class='day-bubble'><div class='day-header'>📅 {day_label}</div>"
                for item in activities:
                    icon = day_map.get(item['type'].lower(), "📍")
                    html += f"<p class='activity'><strong>{item['time']}</strong> — {icon} <strong>{item['type']}</strong>: {item['description']}</p>"
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)


