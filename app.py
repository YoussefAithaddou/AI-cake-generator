import openai
from apikey import key 
import torch
from diffusers import StableDiffusionPipeline 
from PIL import Image
import streamlit as st

#Get the OPENAI's API KEY
openai.api_key=key

torch.cuda.empty_cache()

#Create Stable diffusion Pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, 
                                               safety_checker = None, requires_safety_checker = False)
pipe.to("cuda") 

#A function to get text from OpenAi's 
def get_text(opneAI_object):
    text = list(opneAI_object.choices)[0]
    text=text.to_dict()['message']['content']
    return(text)



#Create Streamlit app: 


# App framework
st.title('🍰 Brand-Inspired Cake Generator: ')
st.write(""" 
AI-Powered Layer Cakes with a Flavor of Your Favorite Brand: with this app, you can turn your favorite brand into a delicious cake.
""")

#Get the brand name
brand= st.text_input('What''s your favorite brand?') 

if brand: 
    #Getdescription of the cake from GPT-3.5:
    text=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"user", "content": f"Imagine {brand} is making a layer cake, write me a detailed description \
            of the product, give this cake a name, your response should be less than or equal to 70 words."}
        ]
    )
    text = get_text(text)

    #Get the stable diffusion prompt:
    prompt=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"user", "content": "write a prompt to create a photo of a layer cake with this description: {text}\
              it is inspired by {brand}'s colors and design, make it precise, full and clear,, your response should\
              be less than or equal to 70 words."}
        ]
    )
    prompt=get_text(prompt)
    #Generate the image and description of the cake:
    image = pipe(prompt).images[0]
    image.save("picture.png")

    #Display the results:
    with st.container():
        image_column, text_column = st.columns((1, 2))
        with image_column:
            st.image(Image.open("picture.png"))
        with text_column:
            st.write(text)
            st.markdown("https://github.com/YoussefAithaddou | youssef.aithddo@gmail.com")