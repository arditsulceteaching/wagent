import requests
from flask import Flask, request, render_template, session, redirect, url_for
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
import uuid
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random string
google_api_key = os.getenv("GOOGLE_API_KEY")
print(google_api_key)

def get_weather(city: str):
    """Get weather for a given city"""
    return {'condition': 'sunny', 'temperature': 25}


def get_location():
    """Get user's current location"""
    from flask import session

    # Check if we have stored location from browser
    if 'user_location' in session:
        lat = session['user_location']['lat']
        lon = session['user_location']['lon']
        print("lat lon", lat, lon)

        # Reverse geocode to get city name using a free API
        try:
            response = requests.get(
                f'https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json',
                headers={'User-Agent': 'WeatherAssistant/1.0'},
                timeout=3
            )
            data = response.json()
            city = data['address'].get('city', data['address'].get('town', 'Unknown'))
            country = data['address'].get('country', '')
            return f"{city}, {country}"
        except:
            return "Rome, Italy"

    return "Rome, Italy"


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
)

system_prompt = """
You are a helpful weather assistant. 
YOUR WORKFLOW:
1. If the user asks about weather WITHOUT specifying a location, you MUST:
   - First call get_location() to find their location
   - Then call get_weather(city) with that location

2. If the user provides a city, call get_weather(city) directly.
"""

# Simple approach - just use the filename directly
memory = SqliteSaver.from_conn_string('checkpoints.db')
checkpointer = memory.__enter__()

agent = create_agent(
    model=llm,
    tools=[get_weather, get_location],
    system_prompt=system_prompt,
    checkpointer=checkpointer
)


@app.route('/')
def home():
    if 'thread_id' not in session:
        session['thread_id'] = str(uuid.uuid4())
    if 'messages' not in session:
        session['messages'] = []

    return render_template('chat.html', messages=session['messages'])


@app.route('/send', methods=['POST'])
def send():
    user_message = request.form.get('message', '').strip()
    user_lat = request.form.get('latitude')
    user_lon = request.form.get('longitude')

    if user_message:
        session['messages'].append({'sender': 'user', 'text': user_message})

        # Store location in session if provided
        if user_lat and user_lon:
            session['user_location'] = {'lat': user_lat, 'lon': user_lon}

        response = agent.invoke(
            {"messages": [{'role': 'user', 'content': user_message}]},
            {"configurable": {"thread_id": session['thread_id']}}
        )

        ai_response = response['messages'][-1].content
        session['messages'].append({'sender': 'agent', 'text': ai_response})
        session.modified = True

    return redirect(url_for('home'))


@app.route('/clear')
def clear():
    session['thread_id'] = str(uuid.uuid4())
    session['messages'] = []
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)