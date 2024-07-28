from gtts import gTTS
import os

# Ensure audio folder exists
audio_folder = os.path.join('assets', 'audio_files')
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

# Words and languages
words = ['hello', 'thank_you', 'i_love_you', 'yes', 'no']
translations = {
    "hello": {"en": "hello", "zh-cn": "你好", "es": "hola", "fr": "bonjour", "de": "hallo", "ja": "こんにちは"},
    "thank_you": {"en": "thank you", "zh-cn": "谢谢", "es": "gracias", "fr": "merci", "de": "danke", "ja": "ありがとう"},
    "i_love_you": {"en": "I love you", "zh-cn": "我爱你", "es": "te quiero", "fr": "je vous aime", "de": "ich liebe dich", "ja": "愛してる"},
    "yes": {"en": "yes", "zh-cn": "是", "es": "sí", "fr": "oui", "de": "ja", "ja": "はい"},
    "no": {"en": "no", "zh-cn": "不", "es": "no", "fr": "non", "de": "nein", "ja": "いいえ"}
}

# Function to generate audio files
def generate_audio_files():
    for word in words:
        for lang_code, translation in translations[word].items():
            tts = gTTS(text=translation, lang=lang_code)
            filename = os.path.join(audio_folder, f"{word}_{lang_code}.mp3")
            tts.save(filename)
            print(f"Audio file generated: {filename}")

# Generate all audio files
generate_audio_files()
