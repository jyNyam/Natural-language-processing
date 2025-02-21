import os
from gtts import gTTS
import pyttsx3

# 1. 텍스트를 음성으로 변환하고 바로 재생하는 함수 (pyttsx3)
def speak_text_with_pyttsx3(text):
    # pyttsx3 초기화
    engine = pyttsx3.init()

    # 음성 속성 설정 (목소리, 속도 등)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # 0: Male, 1: Female (depending on system)
    engine.setProperty('rate', 150)  # 속도 설정
    engine.setProperty('volume', 0.9)  # 볼륨 설정 (0.0 ~ 1.0)

    # 입력된 텍스트를 음성으로 변환 및 재생
    engine.say(text)
    engine.runAndWait()

# 2. 텍스트를 음성으로 변환하고 MP3 파일로 저장하는 함수 (gTTS)
def speak_text_with_gtts(text, mp3_filename="output.mp3"):
    tts = gTTS(text=text, lang='ko')
    
    # MP3 파일로 저장
    tts.save(mp3_filename)
    print(f"Generated speech saved as {mp3_filename}")
    
    # 음성 파일 재생 (플랫폼에 맞게 명령어 변경)
    if os.name == 'nt':  # Windows
        os.system(f'start wmplayer "{mp3_filename}"')
    elif os.name == 'posix':  # macOS or Linux
        os.system(f"open {mp3_filename}" if sys.platform == "darwin" else f"xdg-open {mp3_filename}")

# 사용자로부터 입력을 받아 텍스트 생성
input_text = input("Enter your text: ")

# pyttsx3를 사용하여 입력된 텍스트를 음성으로 변환하고 재생
speak_text_with_pyttsx3(input_text)

# gTTS를 사용하여 입력된 텍스트를 음성으로 변환하고 MP3로 저장 및 재생
speak_text_with_gtts(input_text)