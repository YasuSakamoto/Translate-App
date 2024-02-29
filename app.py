import streamlit as st
#import requests

from main import perform_translation

# タイトル
st.title('English-Japanese Translation')

# 入力プロンプト
st.write('Please enter English')

# Text input for English sentence
english_text = st.text_area("", placeholder="Type here...")
    
# Translate button
if st.button('Translate'):
    # Attempt to perform translation
    translated_text = perform_translation(english_text)
    st.write('翻訳')
    st.text_area("", value=translated_text, height=300, disabled=True)
    