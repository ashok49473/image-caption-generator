# Import necessary modules
import time
import openai
import streamlit as st
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# OpenAI API key
openai.api_key = st.secrets['API_KEY']

# Prompt for the GPT3 model
base_prompt = "rewrite the given image caption in five different ways: \n"

# To get multiple captions using the original caption
def run_gpt3(caption):
    time.sleep(10)
    response = openai.Completion.create(
        model="text-davinci-003", prompt=base_prompt+caption+'.', temperature=0.7, max_tokens=650)
    return response.choices[0].text

# to load caption generation models from hugggingface hub
@st.cache(allow_output_mutation=True)
def get_model():
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
    return processor, model


# model and preprocessor
processor, model = get_model()

st.header("AI tool to create captions for your image!")

# Upload image file
file = st.file_uploader('Upload an image', type=['jpg', 'png'])

# Display uploaded image
if file is not None:
    image = Image.open(file).convert('RGB')
    new_image = image.resize((400, 400))
    st.image(new_image)

# Generate caption and display
if st.button("Get Captions"):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)

    caption = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    st.write("<b>Caption:</b> ", caption, unsafe_allow_html=True)

    # Generate more captions
    alternative_captions = run_gpt3(caption)

    # Display multiple captions
    with st.expander("See more captions.."):
        st.write(alternative_captions)
