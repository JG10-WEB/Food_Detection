import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

model = load_model('ResnetNet_jay.h5')
labels = {0: 'Biryani', 1: 'Chole bature', 2: 'Dhokla', 3: 'Dosa', 4: 'Ghevar', 5: 'Gulab Jamun', 6: 'Halwa', 7: 'Idli', 8: 'Jalebi', 9: 'Kachori', 10: 'Kofta', 11: 'Ladoo', 12: 'Pani puri', 13: 'Paratha', 14: 'Poha', 15: 'Rasgulla', 16: 'Samosa', 17: 'Vada Pav'}

############

##Snacks = ['Vada Pav','Pani Puri','Dhokla','Idli','Kachori','Poha','Samosa','Chole bature','Kofta','Biryani','Dosa','Paratha']
##Sweetmeat = ['Ghevar','Gulab Jamun','Halwa','Jalebi','Ladoo','Rasgulla']



##def fetch_calories(prediction):
    ##try:
        ##url = 'https://www.google.com/search?&q=calories in ' + prediction
        ##req = requests.get(url).text
        ##scrap = BeautifulSoup(req, 'html.parser')
        ##calories = scrap.find("div", class_= "BNeawe iBp4i AP7Wnd").text
        ##return calories
    ##except Exception as e:
        ##st.error("Can't able to fetch the Calories")
        ##print(e)

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()

def run():
    ##st.title("Food Detection")
    st.markdown("""
    <style>
    .big-font {
    font-size:110px !important;
    text-align: center; color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Food Detection</p>', unsafe_allow_html=True)
    ##st.subheader('Kripanshu Ameta(C003)')
    ##st.subheader('Jay Goyal(C017)')
    ##st.subheader('Dhruv Khandelwal(C024)')
    ##st.subheader('Priyansh Tiwari (C043)')
    st.markdown("<h1 style='text-align: center; color: white; font-size:30px !important;'>Kripanshu Ameta(C003)</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white; font-size:30px !important;'>Jay Goyal(C017)</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white; font-size:30px !important;'>Dhruv Khandelwal(C024)</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white; font-size:30px !important;'>Priyansh Tiwari (C043)</h1>", unsafe_allow_html=True)
    
    from PIL import Image
    image = Image.open('food_image_jay.jpg')

    st.image(image, caption='Food brings Eternal bliss')
    
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250,250))
        st.image(img,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            result= processed_img(save_image_path)
            print(result)
            ##if result in Snacks:
                ##st.info('**Category : Snacks**')
            ##else:
                ##st.info('**Category : Sweetmeat**')
            st.success("**Predicted : "+result+'**')
            ##cal = fetch_calories(result)
            ##if cal:
                ##st.warning('**'+cal+'(100 grams)**')
run()