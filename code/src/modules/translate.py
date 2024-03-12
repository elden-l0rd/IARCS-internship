# pip install googletrans==3.1.0a0
from googletrans import Translator

translator = Translator()

def to_not_english(text, target, translator=translator):
    translated = translator.translate(text=text, dest=target)
    # print(translated.text)
    # print(f"Translated to {target}.")
    return translated

def to_english(text, target='en', translator=translator):
    translated = translator.translate(text=text, dest=target)
    # print(f"Translated to {target}.")
    return translated

def back_translation(text, target, translator=translator):
    '''
    text: can be a string or a list of strings
    '''
    not_english = to_not_english(text, target)
    translated = []
    if type(not_english) == list:
        for i in not_english:
            translated.append(to_english(i.text))
    else:
        translated = to_english(not_english.text)
    return translated


# Uncomment everything below to use LibreTranslate API

# """
# LibreTranslate API Usage

# This script utilizes the LibreTranslate API for translating text between languages.
# LibreTranslate offers open-source machine translation, which is free to use under the GNU Affero General Public License v3.0.

# API Reference:
# - GitHub Repository: https://github.com/LibreTranslate/LibreTranslate
# - API Documentation: https://libretranslate.com/docs/

# Please ensure compliance with LibreTranslate's licensing terms and conditions when using their API.

# Dependencies:
# - Requests: Required for making HTTP requests to the LibreTranslate API.

# Usage:
# - The script makes calls to the LibreTranslate API to perform text translation operations.

# License:
# - LibreTranslate is released under the AGPLv3 license. For more details, visit their GitHub repository.

# Note: This documentation is for informational purposes only. Always refer to the official LibreTranslate documentation and license for the most accurate and up-to-date information.
# """

# from libretranslatepy import LibreTranslateAPI
# import urllib.request
# import urllib.error

# def connect_api():
#     urls = ["https://translate.argosopentech.com/", "https://translate.terraprint.co/"]
#     for url in urls:
#         try:
#             urllib.request.urlopen(url)
#             print(f"Successfully connected to {url}\n==========================")
#             return url
#         except urllib.error.HTTPError as e:
#             print(f"HTTPError for {url}: {e.code}")
#         except urllib.error.URLError as e:
#             print(f"URLError for {url}: {e.reason}")
#         except Exception as e:
#             print(f"Error for {url}: {e}")
#     return None

# lt = LibreTranslateAPI(connect_api())

# def back_translation(text, curr, target, lt=lt):
#     '''
#     curr: current language
#     target: target language
#     '''

#     translated = lt.translate(text, curr, target)
#     return translated
