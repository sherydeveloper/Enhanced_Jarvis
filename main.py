import speech_recognition as sr
import webbrowser
import pyttsx3
import musicLibrary
import requests
# from openai import OpenAI
from gtts import gTTS
import pygame
import time
import os
import ai_integration
recognizer = sr.Recognizer()

engine = pyttsx3.init()
apikey = "25913dd79d9b4601aba3a3c3beecbde7"
def speak_old(text):
    engine.say(text)
    engine.runAndWait()


def speak(text):
    tts = gTTS(text)
    tts.save('temp.mp3')
  

    # Initialize Pygame mixer
    pygame.mixer.init()

    # Load the MP3 file
    pygame.mixer.music.load('temp.mp3')  # Replace with the path to your MP3 file

    # Play the MP3 file
    pygame.mixer.music.play()

    # Keep the script running to allow the music to play
    while pygame.mixer.music.get_busy():
        time.sleep(1)

    pygame.mixer.music.unload()  # Replace with the path to your MP3 file
    os.remove('temp.mp3')

    
# def openAI(command):
#     client = OpenAI(
#   api_key="sk-None-qMRNz6mkBqJvCO1WDhUUT3BlbkFJL2Syh1GNusaRU45p640g",)

#     completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a virtual assistant, named Jarvis skilled in general tasks like Alexa and google cloud."},
#         {"role": "user", "content": command}
#     ]
#     )

#     return completion.choices[0].message.content
def processCommand(c):
    if "open google" in c.lower():
        webbrowser.open("https://google.com")
    elif "open facebook" in c.lower():
        webbrowser.open("https://facebook.com")
    elif "open youtube" in c.lower():
        webbrowser.open("https://youtube.com")
    elif "open linkedin" in c.lower():
        webbrowser.open("https://linkedin.com")
    elif "open instagram" in c.lower():
        webbrowser.open("https://instagram.com")
    elif c.lower().startswith("play"):
        song = c.lower().split(" ")[1]
        link = musicLibrary.music[song]
        webbrowser.open(link)

    elif "news" in c.lower():
        r = requests.get(f"https://newsapi.org/v2/top-headlines?country=us&apiKey={apikey}")
        if r.status_code == 200:
            data = r.json()
            
            #Extract Articles

            articles = data.get('articles', [])

            #Headlines

            for article in articles:
                speak(article['title'])

    else:
        #let the openAI handle the command...
        output = ai_integration.get_response(c)
        speak(output)
        

    

if __name__ == "__main__":
    speak("Initializing Jarvis...")
    #Listen the wake Word jarvis
    while True:
        r = sr.Recognizer()

        # recognize speech using Google
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = r.listen(source, timeout= 3, phrase_time_limit= 2)
            print("Recognizing...")
            word = r.recognize_google(audio)
            if(word.lower() == "jarvis"):
                speak("Activating Jarvis, Yes sir how may i help you?")
                #Listen for command
                with sr.Microphone() as source:
                    print("What is Your Command...")
                    audio = r.listen(source, timeout= 3, phrase_time_limit= 2)
                    command = r.recognize_google(audio)
                    # print(command)
                    speak(f"Executing The command {command}")
                    processCommand(command)

            if "jarvis off" in command.lower():
                speak("Have a great day sir Thankyou!, Turning off the program Jarvis")
                break

                

        except Exception as e:
            print("Error; {0}".format(e))
