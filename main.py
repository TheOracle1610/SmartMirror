import sys
import cv2
import requests
import json
import threading
import time
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit,
    QPushButton, QListWidget, QMainWindow, QAction, QMenu, QDialog
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, QRect, QThread, pyqtSignal
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# API Keys
NEWS_API_KEY = "19a6af3a9b824453bc8bdb271ce73eab"
WEATHER_API_KEY = "a0d2b70021dd53af69cf71bf27f1fa62"
GEMINI_API_KEY = "AIzaSyBA_G9uNu-XOMocKy6n3NDgqstRS9J0JO8"

# Initialize speech engine
engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# To-Do List File
TODO_FILE = "todo_list.json"

def load_todo_list():
    """Load the to-do list from a JSON file."""
    if os.path.exists(TODO_FILE):
        with open(TODO_FILE, "r") as file:
            return json.load(file)
    return []

def save_todo_list(todo_list):
    """Save the to-do list to a JSON file."""
    with open(TODO_FILE, "w") as file:
        json.dump(todo_list, file, indent=2)

# Load To-Do List at Startup
todo_list = load_todo_list()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def recognize_speech():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return "Error connecting to speech service."

def manage_todo(command):
    """Manage the to-do list with persistent storage."""
    global todo_list
    todo_list = load_todo_list()

    if "add" in command:
        task = command.replace("add", "").strip()
        todo_list.append(task)
        save_todo_list(todo_list)
        return f"Added '{task}' to your to-do list."

    elif "remove" in command:
        task = command.replace("remove", "").strip()
        if task in todo_list:
            todo_list.remove(task)
            save_todo_list(todo_list)
            return f"Removed '{task}' from your to-do list."
        else:
            return f"'{task}' is not in your to-do list."

    elif "tick" in command or "complete" in command:
        task = command.replace("tick", "").replace("complete", "").strip()
        if task in todo_list:
            task_index = todo_list.index(task)
            todo_list[task_index] = f"✔ {task}"
            save_todo_list(todo_list)
            return f"Marked '{task}' as completed."
        elif f"✔ {task}" in todo_list:
            return f"'{task}' is already marked as completed."
        else:
            return f"'{task}' is not in your to-do list."

    elif "list" in command or "show" in command:
        if todo_list:
            return "Here is your to-do list:\n" + "\n".join(f"- {task}" for task in todo_list)
        else:
            return "Your to-do list is empty."

    else:
        return "I didn't understand that. You can add, remove, tick, or list tasks."

# Fetch Latest News
def get_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    headlines = [article['title'] for article in response.get('articles', [])[:5]]
    return "Here are the latest news headlines:\n" + "\n".join(headlines)

# Fetch Weather Data
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()
    if response.get("cod") != 200:
        return "Could not fetch weather data."
    temp = response["main"]["temp"]
    description = response["weather"][0]["description"]
    return f"The current temperature in {city} is {temp}°C with {description}."

def get_weather2(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()
    if response.get("cod") != 200:
        return "Could not fetch weather data."
    temp = response["main"]["temp"]
    description = response["weather"][0]["description"]
    humidity = response["main"]["humidity"]
    wind_speed = response["wind"]["speed"]
    return {
        "city": city,
        "temperature": temp,
        "description": description,
        "humidity": humidity,
        "wind_speed": wind_speed
    }

# Wikipedia Search
def search_wikipedia(query):
    try:
        return wikipedia.summary(query.lower().replace("tell me","").replace("about",""), sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found: {', '.join(e.options[:3])}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for that topic."

# AI Chatbot using Gemini
def ai_chat(prompt):
    response = model.generate_content(prompt)
    return response.text

def get_weather3(city):
    """Fetch weather details for a given city."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()
    weather = response.get("weather", [{}])[0].get("description", "unknown")
    temp = response.get("main", {}).get("temp", "unknown")
    humidity = response.get("main", {}).get("humidity", "unknown")
    wind_speed = response.get("wind", {}).get("speed", "unknown")
    return weather, temp, humidity, wind_speed

def suggest_clothes(weather, temp, humidity, wind_speed):
    """Use Gemini AI to suggest clothing based on weather conditions."""
    prompt = (
        f"Suggest an outfit based on these conditions:\n"
        f"- Weather: {weather}\n"
        f"- Temperature: {temp}°C\n"
        f"- Humidity: {humidity}%\n"
        f"- Wind Speed: {wind_speed} m/s\n"
    )
    response = model.generate_content(prompt)
    return response.text.replace('*', '')

# NLP-Based Intent Recognition
intent_phrases = {
    "todo": ["to-do", "task", "list", "add", "remove", "tick", "complete"],
    "news": ["news", "headlines", "latest news"],
    "weather": ["weather", "temperature", "forecast"],
        "clothing": ["clothes", "outfit", "wear today"],

    "wiki": ["tell me about"],
    "chat": ["explain", "how does", "why is", "what happens if"]
}

vectorizer = TfidfVectorizer()
vectorizer.fit([" ".join(phrases) for phrases in intent_phrases.values()])

def detect_intent(user_input):
    """Identify intent using TF-IDF & Cosine Similarity."""
    user_vector = vectorizer.transform([user_input])
    intent_vectors = vectorizer.transform([" ".join(phrases) for phrases in intent_phrases.values()])
    similarities = cosine_similarity(user_vector, intent_vectors).flatten()
    if max(similarities) > 0.2:
        return list(intent_phrases.keys())[similarities.argmax()]
    return "unknown"

def get_ip():
    response = requests.get('https://api64.ipify.org?format=json').json()
    return response["ip"]

def get_city_by_ip():
    ip_address = get_ip()
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_data = {
        "ip": ip_address,
        "city": response.get("city"),
        "region": response.get("region"),
        "country": response.get("country_name")
    }
    return location_data["city"] if location_data["city"] else "Unknown"

class TypingEffectThread(QThread):
    update_text = pyqtSignal(str)

    def __init__(self, full_text, delay=0.05):
        super().__init__()
        self.full_text = full_text
        self.delay = delay

    def run(self):
        displayed_text = ""
        for char in self.full_text:
            displayed_text += char
            self.update_text.emit(displayed_text)
            time.sleep(self.delay)

class SmartMirror(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Smart Mirror Simulation")
        self.setGeometry(100, 100, 800, 600)
        self.setFixedSize(800, 600)  # Make the window non-resizable

        # Main widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Camera feed
        self.camera_label = QLabel(self)
        self.camera_label.setGeometry(QRect(0, 0, 800, 600))
        self.camera_label.setStyleSheet("background-color: transparent;")
        self.layout.addWidget(self.camera_label)

        # Weather frame
        self.weather_label = QLabel("Weather", self)
        self.weather_label.setStyleSheet("color: white; font-size: 16px; background-color: transparent;")
        self.weather_label.setAlignment(Qt.AlignLeft)
        self.weather_label.setGeometry(20, 40, 200, 30)

        self.weather_info_label = QLabel("", self)
        self.weather_info_label.setStyleSheet("color: white; font-size: 13px; background-color: transparent;")
        self.weather_info_label.setAlignment(Qt.AlignLeft)
        self.weather_info_label.setWordWrap(True)
        self.weather_info_label.setGeometry(20, 70, 300, 100)

        # Mirror response label
        self.mirror_response_label = QLabel("", self)
        self.mirror_response_label.setStyleSheet("color: white; font-size: 11px; background-color: transparent;")
        self.mirror_response_label.setAlignment(Qt.AlignLeft)
        self.mirror_response_label.setWordWrap(True)
        self.mirror_response_label.setGeometry(20, 180, 500, 180)  # Positioned below weather info

        # To-Do List frame
        self.todo_label = QLabel("To-Do List", self)
        self.todo_label.setStyleSheet("color: white; font-size: 16px; background-color: transparent;")
        self.todo_label.setGeometry(20, 400, 200, 30)

        self.todo_entry = QLineEdit(self)
        self.todo_entry.setStyleSheet("color: white; background-color: transparent; border: 1px solid white;")
        self.todo_entry.setGeometry(20, 430, 200, 30)

        self.todo_list = QListWidget(self)
        self.todo_list.setStyleSheet("color: white; background-color: transparent; border: 1px solid white;")
        self.todo_list.setGeometry(20, 460, 200, 80)

        self.add_button = QPushButton("Add", self)
        self.add_button.setStyleSheet("color: white; background-color: transparent; border: 1px solid white;")
        self.add_button.setGeometry(20, 540, 60, 30)
        self.add_button.clicked.connect(self.add_todo)

        self.remove_button = QPushButton("Remove", self)
        self.remove_button.setStyleSheet("color: white; background-color: transparent; border: 1px solid white;")
        self.remove_button.setGeometry(90, 540, 60, 30)
        self.remove_button.clicked.connect(self.remove_todo)

        self.check_button = QPushButton("Check", self)
        self.check_button.setStyleSheet("color: white; background-color: transparent; border: 1px solid white;")
        self.check_button.setGeometry(160, 540, 60, 30)
        self.check_button.clicked.connect(self.check_todo)

        # News frame
        self.news_label = QLabel("News Headlines", self)
        self.news_label.setStyleSheet("color: white; font-size: 16px; background-color: transparent;")
        self.news_label.setAlignment(Qt.AlignRight)
        self.news_label.setGeometry(570, 400, 200, 30)

        self.news_list = QListWidget(self)
        self.news_list.setStyleSheet("color: white; background-color: transparent; border: 1px solid white;")
        self.news_list.setGeometry(380, 430, 400, 140)
        self.news_list.setWordWrap(True)

        # Time display
        self.time_label = QLabel("", self)
        self.time_label.setStyleSheet("color: white; font-size: 18px; background-color: transparent;")
        self.time_label.setAlignment(Qt.AlignRight)
        self.time_label.setGeometry(580, 40, 200, 30)

        # Menu
        menu = self.menuBar()
        settings_menu = menu.addMenu("Settings")
        open_settings_action = QAction("Open Settings", self)
        open_settings_action.triggered.connect(self.open_settings)
        settings_menu.addAction(open_settings_action)

        # Timer for camera feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_feed)
        self.timer.start(30)

        # Timer for updating time
        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)

        # Fetch city by IP
        self.city = get_city_by_ip()

        # Fetch and display weather data
        self.update_weather_info()

        # Fetch and display news headlines
        news_data = get_news()
        for article in news_data.split("\n")[1:]:
            self.news_list.addItem("● "+article)

       # Load the to-do list from file
        self.load_todo_list()

        # Start news refresh thread
        self.news_thread = threading.Thread(target=self.refresh_news, daemon=True)
        self.news_thread.start()

        # Start chatbot thread
        self.chatbot_thread = threading.Thread(target=self.start_chatbot, daemon=True)
        self.chatbot_thread.start()

    def update_weather_info(self):
        weather_data = get_weather2(self.city)
        if weather_data:
            weather_info = (f"City: {weather_data['city']}\n"
                            f"Temperature: {weather_data['temperature']}°C\n"
                            f"Weather: {weather_data['description']}\n"
                            f"Humidity: {weather_data['humidity']}%\n"
                            f"Wind Speed: {weather_data['wind_speed']} m/s")
            self.weather_info_label.setText(weather_info)
        else:
            self.weather_info_label.setText("Error fetching weather data.")


    def update_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (800, 600))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(img))

    def update_time(self):
        current_time = time.strftime("%H:%M:%S")
        self.time_label.setText(current_time)

    def add_todo(self):
        item = self.todo_entry.text()
        if item:
            self.todo_list.addItem(item)
            self.todo_entry.clear()
            self.save_todo_list()

    def remove_todo(self):
        selected_item = self.todo_list.currentRow()
        if selected_item >= 0:
            self.todo_list.takeItem(selected_item)
            self.save_todo_list()

    def check_todo(self):
        selected_item = self.todo_list.currentRow()
        if selected_item >= 0:
            item = self.todo_list.takeItem(selected_item)
            item.setText(f"[✔] {item.text()}")
            self.todo_list.insertItem(selected_item, item)
            self.save_todo_list()

    def save_todo_list(self):
        tasks = [self.todo_list.item(i).text() for i in range(self.todo_list.count())]
        with open("todo_list.json", "w") as file:
            json.dump(tasks, file)

    def load_todo_list(self):
        try:
            with open("todo_list.json", "r") as file:
                tasks = json.load(file)
                for task in tasks:
                    self.todo_list.addItem(task)
        except FileNotFoundError:
            pass

    def refresh_news(self):
        while True:
            news_data = get_news()
            self.news_list.clear()
            for article in news_data.split("\n")[1:]:
                self.news_list.addItem(article)
            time.sleep(300)  # Refresh every 5 minutes

    def open_settings(self):
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("Settings")
        self.settings_dialog.setGeometry(200, 200, 300, 100)

        layout = QVBoxLayout()
        label = QLabel("City for Weather:", self)
        layout.addWidget(label)

        self.city_entry = QLineEdit(self)
        self.city_entry.setText(self.city)
        layout.addWidget(self.city_entry)

        save_button = QPushButton("Save", self)
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)

        self.settings_dialog.setLayout(layout)
        self.settings_dialog.exec_()

    def save_settings(self):
        self.city = self.city_entry.text()
        self.update_weather_info()
        self.settings_dialog.close()
        
    def start_chatbot(self):
        speak("Smart Mirror AI is ready!\n")
        while True:
            user_input = recognize_speech()
            print(user_input)
            if not user_input.startswith("mirror"):
                continue
            user_input = user_input.replace("mirror", "").strip()
            intent = detect_intent(user_input)
            weather, temp, humidity, wind_speed = get_weather3(self.city)

            if "weather" in user_input.lower():
                intent = "weather"
                
            if intent == "todo":
                response = manage_todo(user_input)
                self.todo_entry.clear()
                load_todo_list()
            elif intent == "clothing":
                response = suggest_clothes(weather, temp, humidity, wind_speed)
            elif intent == "news":
                response = "\n".join([self.news_list.item(i).text() for i in range(self.news_list.count())])
            elif intent == "weather":
                response = self.weather_info_label.text()
            elif intent == "wiki":
                response = search_wikipedia(user_input)
            elif intent == "chat":
                response = ai_chat(user_input)
            else:
                response = "I didn't understand that. Try again."

            threading.Thread(target=self.display_typing_effect, args=(response,), daemon=True).start()
            speak(response)

    def display_typing_effect(self, text):
        self.mirror_response_label.setText("")
        for i in range(len(text) + 1):
            self.mirror_response_label.setText(text[:i])
            QApplication.processEvents()
            time.sleep(0.05)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mirror = SmartMirror()
    mirror.cap = cv2.VideoCapture(0)
    mirror.show()
    sys.exit(app.exec_())
