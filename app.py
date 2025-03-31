import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from datetime import date
from urllib.parse import quote_plus
from io import BytesIO
from fpdf import FPDF
import re

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AI Travel Agent", page_icon="🌍", layout="wide")
st.title("🌍 AI Travel Agent")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_form_panel" not in st.session_state:
    st.session_state.show_form_panel = True

# Model
chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.9,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Sidebar
with st.sidebar:
    if st.button("🧾 Toggle Travel Preferences Panel"):
        st.session_state.show_form_panel = not st.session_state.show_form_panel

    if st.session_state.show_form_panel:
        st.header("🧾 Travel Preferences")

        if "trip_generated" not in st.session_state:
            destination = st.text_input("Where would you like to travel?")
            start_date = st.date_input("Start date", date.today())
            end_date = st.date_input("End date", date.today())
            budget = st.selectbox("What is your budget level?", ["Low", "Medium", "High"])
            interests = st.text_area("Main interests (e.g., food, museums, nature, nightlife)")
            avoid = st.text_input("Anything to avoid?")

            if st.button("🧽 Generate Trip Plan"):
                if not destination or not interests:
                    st.warning("Please fill out at least the destination and interests.")
                else:
                    with st.spinner("Generating your custom itinerary..."):
                        st.session_state.destination = destination
                        st.session_state.start_date = start_date
                        st.session_state.end_date = end_date
                        st.session_state.budget = budget
                        st.session_state.interests = interests
                        st.session_state.avoid = avoid

                        trip_days = (end_date - start_date).days + 1
                        prompt = f"""
You are a smart travel planner that generates personalized travel itineraries.

Create a personalized day-by-day travel itinerary:
Destination: {destination}
Trip Duration: {trip_days} days
Start Date: {start_date}
End Date: {end_date}
Budget Level: {budget}
Interests: {interests}
Things to Avoid: {avoid}

Use the following format strictly for each day:
Day X: Title of the Day
[HH:MM] Activity description | Status: Done/In Progress/Not Yet Started
"""
                        st.session_state.chat_history.append(("User", prompt))
                        messages = [
                            HumanMessage(content=msg) if role == "User" else AIMessage(content=msg)
                            for role, msg in st.session_state.chat_history
                        ]
                        response = chat.invoke(messages)
                        st.session_state.chat_history.append(("AI", response.content))
                        st.session_state.trip_generated = True
                        st.rerun()

# Display itinerary
if st.session_state.get("trip_generated"):

    st.markdown("### 📋 Initial Trip Plan")
    if len(st.session_state.chat_history) > 1:
        def parse_itinerary(text):
            itinerary = []
            current_day = None
            for line in text.split("\n"):
                day_match = re.match(r"Day (\d+):? ?(.*)?", line.strip())
                activity_match = re.match(r"\[(\d{1,2}:\d{2})\] (.*?) \| Status: (.*)", line.strip())
                if day_match:
                    current_day = {"day": f"Day {day_match.group(1)}", "title": day_match.group(2), "activities": []}
                    itinerary.append(current_day)
                elif activity_match and current_day:
                    current_day["activities"].append({
                        "time": activity_match.group(1),
                        "desc": activity_match.group(2),
                        "status": activity_match.group(3)
                    })
            return itinerary

    st.markdown("### 💬 Chat with Your AI Travel Agent")
    for speaker, msg in st.session_state.chat_history[1:]:
        if speaker == "AI" and msg.startswith("Day"):
            itinerary = parse_itinerary(msg)
            for day in itinerary:
                bubble_style = "background-color:rgba(240,240,240,0.7); color:#333; padding:10px; border-radius:10px; margin:5px;"
                st.markdown(f"<div style='{bubble_style}'>", unsafe_allow_html=True)
                st.markdown(f"**{day['day']}: {day['title']}**")
                table_md = "| Time | Activity | Status |\n|------|----------|--------|"
                for act in day["activities"]:
                    table_md += f"\n| {act['time']} | {act['desc']} | {act['status']} |"
                st.markdown(table_md)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            if speaker == "User":
                bubble_style = "background-color:rgba(224,247,250,0.7); color:#000;"
            else:
                bubble_style = "background-color:rgba(240,240,240,0.7); color:#000;"
            st.markdown(f"<div style='{bubble_style} padding:10px; border-radius:10px; margin:5px;'>{msg}</div>", unsafe_allow_html=True)

    followup = st.chat_input("Ask to adjust your trip:")
    if followup:
        with st.spinner("Updating your itinerary..."):
            st.session_state.chat_history.append(("User", followup))
            messages = [
                HumanMessage(content=msg) if role == "User" else AIMessage(content=msg)
                for role, msg in st.session_state.chat_history
            ]
            response = chat.invoke(messages)
            st.session_state.chat_history.append(("AI", response.content))
            st.rerun()


    st.markdown("---")
    st.markdown("### 📄 Download Your Itinerary as PDF")
    if st.button("⬇️ Download PDF"):

        class PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 16)
                self.set_text_color(0, 102, 204)
                self.cell(0, 10, "Daily Itinerary", ln=True, align="C")
                self.ln(5)

        def clean_text(text):
            replacements = {
                "–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
                "•": "*", "…": "...", "é": "e", "á": "a", "œ": "oe"
            }
            for bad, good in replacements.items():
                text = text.replace(bad, good)
            return text

        def parse_itinerary(text):
            itinerary = []
            current_day = None
            for line in text.split("\n"):
                day_match = re.match(r"Day (\d+):? ?(.*)?", line.strip())
                activity_match = re.match(r"\[(\d{1,2}:\d{2})\] (.*?) \| Status: (.*)", line.strip())
                if day_match:
                    current_day = {"day": f"Day {day_match.group(1)}", "title": day_match.group(2), "activities": []}
                    itinerary.append(current_day)
                elif activity_match and current_day:
                    current_day["activities"].append({
                        "time": activity_match.group(1),
                        "desc": activity_match.group(2),
                        "status": activity_match.group(3)
                    })
            return itinerary

        def status_color(status):
            return {
                "Done": (200, 255, 200),
                "In Progress": (255, 255, 180),
                "Not Yet Started": (255, 220, 220)
            }.get(status, (240, 240, 240))

        itinerary_text = st.session_state.chat_history[1][1]
        itinerary = parse_itinerary(st.session_state.itinerary_text)
        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", size=12)


        for day in itinerary:
            pdf.add_page()  # Start each day on a new page
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(0)
            pdf.cell(0, 10, clean_text(f"{day['day']}: {day['title']}"), ln=True)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_fill_color(200, 200, 200)
            pdf.cell(40, 10, "Time", border=1, fill=True)
            pdf.cell(100, 10, "Activity", border=1, fill=True)
            pdf.cell(50, 10, "Status", border=1, ln=True, fill=True)
            pdf.set_font("Helvetica", size=12)

            for act in day["activities"]:
                r, g, b = status_color(act["status"])
                pdf.set_fill_color(r, g, b)

                x = pdf.get_x()
                y = pdf.get_y()
                line_height = 7
                col_widths = [40, 100, 50]

                # Use dummy PDF to calculate height
                dummy = FPDF()
                dummy.add_page()
                dummy.set_font("Helvetica", size=12)
                time_lines = dummy.multi_cell(col_widths[0], line_height, clean_text(act["time"]), split_only=True)
                desc_lines = dummy.multi_cell(col_widths[1], line_height, clean_text(act["desc"]), split_only=True)
                status_lines = dummy.multi_cell(col_widths[2], line_height, clean_text(act["status"]), split_only=True)
                max_lines = max(len(time_lines), len(desc_lines), len(status_lines))
                row_height = max_lines * line_height

                pdf.set_xy(x, y)
                pdf.cell(col_widths[0], row_height, clean_text(act["time"]), border=1, fill=True)
                pdf.set_xy(x + col_widths[0], y)
                pdf.multi_cell(col_widths[1], line_height, clean_text(act["desc"]), border=1, fill=True)
                y_after_desc = pdf.get_y()
                pdf.set_xy(x + col_widths[0] + col_widths[1], y)
                pdf.cell(col_widths[2], row_height, clean_text(act["status"]), border=1, fill=True)
                pdf.set_y(max(y_after_desc, y + row_height))

            pdf.ln(5)

        output = pdf.output(dest="S")
        if isinstance(output, str):
            output = output.encode("latin1")
        buffer = BytesIO(output)
        buffer.seek(0)

        st.download_button(
            label="Download Itinerary PDF",
            data=buffer,
            file_name="trip_itinerary.pdf",
            mime="application/pdf"
        )
