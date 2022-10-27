import streamlit as st
from PIL import Image
import os
import torch

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration
)
from torch import cuda
#from src.utils.utils import *

device = torch.device('cuda:0' if cuda.is_available() else 'cpu')

t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')

t5_model_dir=os.path.join('model/t5_20/t5-large')
t5_model= T5ForConditionalGeneration.from_pretrained(t5_model_dir)
t5_model.to(device)

st.title('Transfer Market News Generation')
col1,col2 = st.columns(2)

with col1:
    st.subheader('Inputs')

    player = st.text_input('Player Name', 'Cristiano Ronaldo')
    team = st.text_input('Club Name', 'Manchester United')
    transfer = st.text_input('Transfer Market Term', 'For Sale')
    league=st.text_input('League', 'Premier League')
    team_b = st.text_input('TeamB Name', 'Paris Saint Germain')

    st.image(Image.open('deltatre-innovation-lab-logo_RGB_deltatre-red.png'), width=250)


with col2:
    
    st.subheader('Model Input String')

    input_text=f'<|PERSON|> {player} <|PERSON|> <|CLUB|>{team}<|CLUB|> <|TRANSFER_MARKET|> {transfer} <|TRANSFER_MARKET|> <|COMPETITION|> {league} <|COMPETITION|> <|CLUB|> {team_b} <|CLUB|>'
    st.write(input_text)

    input_ids = t5_tokenizer(input_text, return_tensors="pt")
    input_ids = input_ids.to(device)

    generated_ids = t5_model.generate(input_ids = input_ids['input_ids'], attention_mask = input_ids['attention_mask'], max_length=256)
    preds = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    st.subheader('Output')
    output_string=' '.join(preds)
    st.write(output_string)