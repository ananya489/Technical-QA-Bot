import json
from pathlib import Path

import streamlit as st

from config import DATA_DIR, MODEL_DIR
from src.pipeline import TechnicalQABot
from src.qa_model import ExtractiveQAModel

st.set_page_config(page_title='Technical Documentation QA Bot', page_icon='🎓', layout='centered')


@st.cache_data
def load_manual():
    return json.loads((DATA_DIR / 'university_manual.json').read_text(encoding='utf-8'))


@st.cache_resource
def load_bot():
    reader = ExtractiveQAModel()
    try:
        reader.load()
        model_status = 'RoBERTa model loaded'
    except Exception:
        model_status = 'RoBERTa unavailable — using lightweight extractive fallback'
    bot = TechnicalQABot(load_manual(), reader=reader, use_model=False)
    return bot, model_status


st.title('Closed-Domain Technical Documentation QA Bot')
st.caption('Answers are restricted to the provided university technical manual.')

bot, status = load_bot()
st.info(status)

question = st.text_input('Ask a question', placeholder='e.g. What is the minimum attendance required?')

if st.button('Get Answer', type='primary'):
    result = bot.ask(question)
    st.subheader('Answer')
    st.write(result['answer'])

    if result['section']:
        st.subheader('Source')
        st.write(result['section'].replace('_', ' ').title())
        st.caption(f"Retrieval score: {result['retrieval_score']:.3f} | Reader confidence: {result['confidence']:.3f}")
        with st.expander('Retrieved context'):
            st.write(result['context'])

with st.expander('Available manual sections'):
    for name in load_manual():
        st.write('• ' + name.replace('_', ' ').title())
