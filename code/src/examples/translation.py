import sys
import os
# Add the directory above "examples" (i.e., "src") to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.translate import back_translation

text = "Hello world!"
# text = ["Hello world!", "How are you ?", "Do you speak english ?", "Goodbye"]
target = "de" # German
translated_text = back_translation(text=text, target=target)

if type(translated_text) == list:
    for i in translated_text:
        print(i.text)
else:
    print(translated_text.text)


# Uncomment everything below to use LibreTranslate API

# # refer to translate.py for more documentation

# text = "Hello world!"
# curr = "en"
# target  ="de" # German
# translated_text = back_translation(text=text,
#                                    curr=curr,
#                                    target=target)
# print(translated_text)
