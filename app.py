import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("pnadel/love-poems")
    model = AutoModelForCausalLM.from_pretrained("pnadel/love-poems")
    return tokenizer, model
tokenizer, model = get_model()

st.title(':heart: Love Poem Generator :heart:')
st.write(
    """
    Love is in the air at the Tufts Medford/Somerville campus! Use this Love Poem Generator fine-tuned 
    from [OpenAI's GPT2 model](https://huggingface.co/gpt2) to compose a Valentine's Day aubade for a lover. 
    You can supply a prompt or get started with nothing. 
    """
)

prompt = st.text_input('Enter a prompt or leave this blank', '')
prompt = '<|startoftext|>' + prompt 

poem_amount = st.number_input('How many poems do you want to generate?', value=3)
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

outputs = model.generate(
    generated,
    do_sample=True,
    top_k=50,
    max_length=150,
    top_p=.95,
    num_return_sequences=poem_amount
)

for i, output in enumerate(outputs):
    st.markdown(f"**Poem {i+1}**\n")
    st.text(tokenizer.decode(output, skip_special_tokens=True))
    st.markdown("<hr style='width: 75%;margin: auto;'>",unsafe_allow_html=True)

st.write("Not happy with the results? Try running it again!")
if st.button('Rerun'):
    st.experimental_rerun()
st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)