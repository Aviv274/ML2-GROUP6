import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import datetime
from agent import Agent
import json
import re
import base64


# Load environment variables
load_dotenv()

st.set_page_config(page_title="Trip Genie", layout="wide")

# Constants
CURRENT_YEAR = datetime.datetime.now().year
response =""


#Button state
if "start_clicked" not in st.session_state:
    st.session_state.start_clicked = False

# Background Image function
def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Show background image until user clicks in the start button
if not st.session_state.start_clicked:
    background_base64 = get_base64_image("images/Trip_Genie.png")

    page_bg = f"""
    <style>
    [data-testid="stApp"] {{
        background-image: url("data:image/png;base64,{background_base64}");
        background-size: cover;
        background-position: center;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

    # Centered button
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.markdown("##")
    st.markdown("##")
    st.markdown("##")
    st.markdown("##")
    st.markdown("##")
    

    col1, col2, col3 = st.columns([2, 2, 0.6])
    with col2:
        if st.button("Start Planning ✈️", key="start_button"):
            st.session_state.start_clicked = True

        
    
else:

    # Instantiate the agent
    agent = Agent()
    response = ""

    if "show_form" not in st.session_state:
        st.session_state.show_form = True

    page_bg = f"""
    <style>
    [data-testid="stApp"] {{
        background-image: url("https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8dHJhdmVsfGVufDB8fDB8fHww");
        background-size: cover;
        background-position: center;
        color : black;
        text-align: center;
        border-radius : 15px
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)




    st.title("Plan Your Dream Trip")

    st.markdown("")

    with st.expander("✏️ Trip Preferences", expanded=st.session_state.show_form):

        expander_bg = f"""
        <style>
        [data-testid="stExpander"] {{
            background-color : white;
            text-align: center;
            border-radius : 15px
        }}
        
        </style>
        """
        st.markdown(expander_bg, unsafe_allow_html=True)

        st.markdown("")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            origin = st.text_input("Origin 🏠")

        with col2:
            destination = st.text_input("Destination 🌆")

        with col3:
            start_date = st.date_input("Start date 📅", datetime.date.today())

        with col4:
            end_date = st.date_input("End date 📅", datetime.date.today())

        
        col5, col6, col7 = st.columns(3)

        with col5:
            adult = st.number_input("Number of adults 🧑 ", min_value=0, value=1)

        with col6:
            children = st.number_input("Number of children 👶 ", min_value=0, value=0)

        with col7:
            budget = st.selectbox("Budget Level 💰", ["Low", "Medium", "High"])

        col8, col9 = st.columns(2)

        with col8:
            interests = st.text_area("Main interests ✨ (e.g., food, museums, nature, nightlife)")

        with col9:
            avoid = st.text_area("Anything to avoid? 🚫")

        st.markdown("</div>", unsafe_allow_html=True)
    

    if st.button("✈️ Generate Itinerary",use_container_width=True):
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

        #st.session_state.show_form = False



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
    st.markdown("###")
   
    st.markdown(f"""
        <div style="
            background-color: rgba(255, 255, 255, 0.60);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            text-align: center;
        ">
            <h3 style="margin: 0;">{response['general']}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("###")

    hcol,space, fcol, space2, gcol = st.columns([2,0.1,2,0.1,2])
    with hcol:
        hotel = response.get("hotel", {})
        link = hotel.get("link")

        if not link and hotel.get("name"):
           link = f"https://www.google.com/search?q={hotel['name'].replace(' ', '+')}+booking"

        st.markdown(f"""
            <div class='card' style="
                background-color: rgba(255, 255, 255, 0.80);
                border-radius: 15px;
                padding: 1em;
                text-align: center;
                min-height: 230px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ">
                <div>
                    <h4>🏨 Hotel </h4>
                    <p><strong>{response['hotel']['name']}</strong><br>
                    {response['hotel']['price_per_night']} per night<br>
                    ⭐ {response['hotel']['rating']}</p>
                </div>
                <a href="{response['hotel']['link']}" target="_blank">Hotel Website</a>
            </div>
        """, unsafe_allow_html=True)

    with fcol:
        flight = response.get("flight", {})

        if "outbound" in flight:
            outbound = flight["outbound"]
            airline = outbound.get('airline', 'N/A')
            departure_time = outbound.get('departure_time', 'N/A')
            arrival_time = outbound.get('arrival_time', 'N/A')
            departure_airport = outbound.get('departure_airport', 'N/A')
            arrival_airport = outbound.get('arrival_airport', 'N/A')
            price = outbound.get('price', 'N/A')
        
            st.markdown(f"""
    <div class='card' style="
        background-color: rgba(255, 255, 255, 0.85);
        padding: 1.5rem;
        border-radius: 12px;
        min-height: 230px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;">
        <h4>✈️ Outbound Flight</h4>
        <p>
            <strong>{airline}</strong><br>
            {departure_time} → {arrival_time}<br>
            {departure_airport} → {arrival_airport}<br>
            💰 {price}<br>
        </p>
    </div>
    """, unsafe_allow_html=True)

    with gcol:
        flight = response.get("flight", {})
        
        if "return" in flight:
            returning = flight["return"]
            airline = returning.get('airline', 'N/A')
            departure_time = returning.get('departure_time', 'N/A')
            arrival_time = returning.get('arrival_time', 'N/A')
            departure_airport = returning.get('departure_airport', 'N/A')
            arrival_airport = returning.get('arrival_airport', 'N/A')
            price = returning.get('price', 'N/A')
     
  
            st.markdown(f"""
    <div class='card' style="
        background-color: rgba(255, 255, 255, 0.85);
        padding: 1.5rem;
        min-height: 230px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h4>🔁 Return Flight</h4>
        <p>
            <strong>{airline}</strong><br>
            {departure_time} → {arrival_time}<br>
            {departure_airport} → {arrival_airport}<br>
            💰 {price}<br>
        </p>
    </div>
    """, unsafe_allow_html=True)
        

    st.markdown("###")

    st.markdown("<h3 style='text-align: center;'>🗓️ Itinerary</h3>", unsafe_allow_html=True)
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
                html = f"""
            <div class='day-bubble' style="
                background-color: rgba(255, 255, 255, 0.80);
                font-weight: bold;
                text-align: center;
                border-radius: 8px;
                padding: 1em;
                margin-bottom: 1.5em;
            ">
                <div class='day-header'>📅 {day_label}</div>
            """
                for item in activities:
                    icon = day_map.get(item['type'].lower(), "📍")
                    html += f"<p class='activity'><strong>{item['time']}</strong> — {icon} <strong>{item['type']}</strong>: {item['description']}</p>"
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)
