import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

model1 = load_model('ResnetNet_jay.h5')
model2 = load_model('mobilenet_jay.h5')
#model3 = load_model('model_keras_improv_2_addition_jay.h5')
model4 = load_model('InceptionNet_new.h5')
labels = {0: 'Biryani-(600 calories per plate)- Ingredients- Chilli Powder , Turmeric , Peas , Onion , Green Chillies, Garam Masala', 1: 'Chole bature-(427 calories per serving)- Ingredients- Cholle masala, semolina, baking soda, Cumin, Cinnamon, Red Mirchi, Chole', 2: 'Dhokla-(75 calories per piece)- Ingredients- Besan, Oil, Mustard Seeds, Lemon Juice, Baking Soda , Curry Leaves, Corriander, Green Chilli', 3: 'Dosa-(113 calories per dosa)- Ingredients- Rice Batter, Semolina, Urad dal, Fenugreek Seeds', 4: 'Ghevar-(74 calories per 100 grams)- Ingredients- Ghee, Sugar, Saffron, Maida, Pistachio, Flour, Milk', 5: 'Gulab Jamun-(432 calories for 100 grams)- Ingredients- Milk, Yoghurt, Butter, Baking Soda, Saffron, Cardaman Powder', 6: 'Halwa-(469 calories per 100 grams)- Ingredients- Carrot, Sugar, Mawa, Ghee, Cardaman, Almonds', 7: 'Idli-(35-39 calories for medium size)- Ingredients- Baking Powder, Butter, Yoghurt, Chilli, Rice, Urad Dal', 8: 'Jalebi-(150 calories per piece)- Ingredients- Sugar, Yoghurt, Maida, Ghee, Saffron, Cardaman Powder, Baking Powder', 9: 'Kachori-(195 calories per piece)- Ingredients- Moong Dal, Maida, Ghee, Cumin,Corriander,Mango,Corriander Powder, Fennel, Garam Masala', 10: 'Kofta-(77 calories per serving)- Ingredients- Paneer, Potato, Peas, Milk, Garam Masala, Chilli and Corriander Powder, Corn Starch', 11: 'Ladoo-(204 calories per piece)- Ingredients- Besan, Ghee, Baking Soda, Green Cardamom, Edible Food Color', 12: 'Pani puri-(329 calories per serving)- Ingredients- Puri/Golgappa, Tamarind Chutney, Sev, Corriander/Mint, Potato, Moong, Boondi', 13: 'Paratha-(126 calories per piece)- Ingredients- Potato, Whole wheat, Flour, Ghee, Chilli Pepper, lemon, Cabbbage, Methi', 14: 'Poha-(158 calories per cup)- Ingredients- Poha, lemon, Green chilli, Onion, Potato, Corriander', 15: 'Rasgulla-(120 calories per piece)- Ingredients- Maida, Chhena, Sugar, Milk', 16: 'Samosa-(262 calories per piece)- Ingredients- Potato, Peas, Garam Masala, Wheat flour', 17: 'Vada Pav-(140 calories per piece)- Ingredients- Potato, Green Chilli, Corriander, Garlic, Channa Flour, Bread, Soda'}
#labels = {0: 'Biryani-(600 calories per plate)', 1: 'Chole bature-(427 calories per serving)', 2: 'Dhokla-(75 calories per piece)', 3: 'Dosa-(113 calories per dosa)', 4: 'Ghevar-(74 calories per 100 grams)', 5: 'Gulab Jamun-(432 calories for 100 grams)', 6: 'Halwa-(469 calories per 100 grams)', 7: 'Idli-(35-39 calories for medium size)', 8: 'Jalebi-(150 calories per piece)', 9: 'Kachori-(195 calories per piece)', 10: 'Kofta-(77 calories per serving)', 11: 'Ladoo-(204 calories per piece)', 12: 'Pani puri-(329 calories per serving)', 13: 'Paratha-(126 calories per piece)', 14: 'Poha-(158 calories per cup)', 15: 'Rasgulla-(120 calories per piece)', 16: 'Samosa-(262 calories per piece)', 17: 'Vada Pav-(140 calories per piece)'}
#labels2 = {0: 'Biryani-(600 calories per plate)', 1: 'Chole bature-(427 calories per serving)', 2: 'Dhokla-(75 calories per piece)', 3: 'Dosa-(113 calories per dosa)', 4: 'Ghevar-(74 calories per 100 grams)', 5: 'Gulab Jamun-(432 calories for 100 grams)', 6: 'Halwa-(469 calories per 100 grams)', 7: 'Idli-(35-39 calories for medium size)', 8: 'Jalebi-(150 calories per piece)', 9: 'Kachori-(195 calories per piece)', 10: 'Kofta-(77 calories per serving)', 11: 'Ladoo-(204 calories per piece)', 12: 'Pani puri-(329 calories per serving)', 13: 'Paratha-(126 calories per piece)', 14: 'Poha-(158 calories per cup)', 15: 'Rasgulla-(120 calories per piece)', 16: 'Samosa-(262 calories per piece)', 17: 'Vada Pav-(140 calories per piece)'}
labels2 = {0: 'Biryani-(600 calories per plate)- Ingredients- Chilli Powder , Turmeric , Peas , Onion , Green Chillies, Garam Masala', 1: 'Chole bature-(427 calories per serving)- Ingredients- Cholle masala, semolina, baking soda, Cumin, Cinnamon, Red Mirchi, Chole', 2: 'Dhokla-(75 calories per piece)- Ingredients- Besan, Oil, Mustard Seeds, Lemon Juice, Baking Soda , Curry Leaves, Corriander, Green Chilli', 3: 'Dosa-(113 calories per dosa)- Ingredients- Rice Batter, Semolina, Urad dal, Fenugreek Seeds', 4: 'Ghevar-(74 calories per 100 grams)- Ingredients- Ghee, Sugar, Saffron, Maida, Pistachio, Flour, Milk', 5: 'Gulab Jamun-(432 calories for 100 grams)- Ingredients- Milk, Yoghurt, Butter, Baking Soda, Saffron, Cardaman Powder', 6: 'Halwa-(469 calories per 100 grams)- Ingredients- Carrot, Sugar, Mawa, Ghee, Cardaman, Almonds', 7: 'Idli-(35-39 calories for medium size)- Ingredients- Baking Powder, Butter, Yoghurt, Chilli, Rice, Urad Dal', 8: 'Jalebi-(150 calories per piece)- Ingredients- Sugar, Yoghurt, Maida, Ghee, Saffron, Cardaman Powder, Baking Powder', 9: 'Kachori-(195 calories per piece)- Ingredients- Moong Dal, Maida, Ghee, Cumin,Corriander,Mango,Corriander Powder, Fennel, Garam Masala', 10: 'Kofta-(77 calories per serving)- Ingredients- Paneer, Potato, Peas, Milk, Garam Masala, Chilli and Corriander Powder, Corn Starch', 11: 'Ladoo-(204 calories per piece)- Ingredients- Besan, Ghee, Baking Soda, Green Cardamom, Edible Food Color', 12: 'Pani puri-(329 calories per serving)- Ingredients- Puri/Golgappa, Tamarind Chutney, Sev, Corriander/Mint, Potato, Moong, Boondi', 13: 'Paratha-(126 calories per piece)- Ingredients- Potato, Whole wheat, Flour, Ghee, Chilli Pepper, lemon, Cabbbage, Methi', 14: 'Poha-(158 calories per cup)- Ingredients- Poha, lemon, Green chilli, Onion, Potato, Corriander', 15: 'Rasgulla-(120 calories per piece)- Ingredients- Maida, Chhena, Sugar, Milk', 16: 'Samosa-(262 calories per piece)- Ingredients- Potato, Peas, Garam Masala, Wheat flour', 17: 'Vada Pav-(140 calories per piece)- Ingredients- Potato, Green Chilli, Corriander, Garlic, Channa Flour, Bread, Soda'}
############
#labels3 = {0: 'Biryani-(600 calories per plate)- Ingredients- Chilli Powder , Turmeric , Peas , Onion , Green Chillies, Garam Masala', 1: 'Chole bature-(427 calories per serving)- Ingredients- Cholle masala, semolina, baking soda, Cumin, Cinnamon, Red Mirchi, Chole', 2: 'Dhokla-(75 calories per piece)- Ingredients- Besan, Oil, Mustard Seeds, Lemon Juice, Baking Soda , Curry Leaves, Corriander, Green Chilli', 3: 'Dosa-(113 calories per dosa)- Ingredients- Rice Batter, Semolina, Urad dal, Fenugreek Seeds', 4: 'Ghevar-(74 calories per 100 grams)- Ingredients- Ghee, Sugar, Saffron, Maida, Pistachio, Flour, Milk', 5: 'Gulab Jamun-(432 calories for 100 grams)- Ingredients- Milk, Yoghurt, Butter, Baking Soda, Saffron, Cardaman Powder', 6: 'Halwa-(469 calories per 100 grams)- Ingredients- Carrot, Sugar, Mawa, Ghee, Cardaman, Almonds', 7: 'Idli-(35-39 calories for medium size)- Ingredients- Baking Powder, Butter, Yoghurt, Chilli, Rice, Urad Dal', 8: 'Jalebi-(150 calories per piece)- Ingredients- Sugar, Yoghurt, Maida, Ghee, Saffron, Cardaman Powder, Baking Powder', 9: 'Kachori-(195 calories per piece)- Ingredients- Moong Dal, Maida, Ghee, Cumin,Corriander,Mango,Corriander Powder, Fennel, Garam Masala', 10: 'Kofta-(77 calories per serving)- Ingredients- Paneer, Potato, Peas, Milk, Garam Masala, Chilli and Corriander Powder, Corn Starch', 11: 'Ladoo-(204 calories per piece)- Ingredients- Besan, Ghee, Baking Soda, Green Cardamom, Edible Food Color', 12: 'Pani puri-(329 calories per serving)- Ingredients- Puri/Golgappa, Tamarind Chutney, Sev, Corriander/Mint, Potato, Moong, Boondi', 13: 'Paratha-(126 calories per piece)- Ingredients- Potato, Whole wheat, Flour, Ghee, Chilli Pepper, lemon, Cabbbage, Methi', 14: 'Poha-(158 calories per cup)- Ingredients- Poha, lemon, Green chilli, Onion, Potato, Corriander', 15: 'Rasgulla-(120 calories per piece)- Ingredients- Maida, Chhena, Sugar, Milk', 16: 'Samosa-(262 calories per piece)- Ingredients- Potato, Peas, Garam Masala, Wheat flour', 17: 'Vada Pav-(140 calories per piece)- Ingredients- Potato, Green Chilli, Corriander, Garlic, Channa Flour, Bread, Soda'}
labels4 = {0: 'Biryani-(600 calories per plate)- Ingredients- Chilli Powder , Turmeric , Peas , Onion , Green Chillies, Garam Masala', 1: 'Chole bature-(427 calories per serving)- Ingredients- Cholle masala, semolina, baking soda, Cumin, Cinnamon, Red Mirchi, Chole', 2: 'Dhokla-(75 calories per piece)- Ingredients- Besan, Oil, Mustard Seeds, Lemon Juice, Baking Soda , Curry Leaves, Corriander, Green Chilli', 3: 'Dosa-(113 calories per dosa)- Ingredients- Rice Batter, Semolina, Urad dal, Fenugreek Seeds', 4: 'Ghevar-(74 calories per 100 grams)- Ingredients- Ghee, Sugar, Saffron, Maida, Pistachio, Flour, Milk', 5: 'Gulab Jamun-(432 calories for 100 grams)- Ingredients- Milk, Yoghurt, Butter, Baking Soda, Saffron, Cardaman Powder', 6: 'Halwa-(469 calories per 100 grams)- Ingredients- Carrot, Sugar, Mawa, Ghee, Cardaman, Almonds', 7: 'Idli-(35-39 calories for medium size)- Ingredients- Baking Powder, Butter, Yoghurt, Chilli, Rice, Urad Dal', 8: 'Jalebi-(150 calories per piece)- Ingredients- Sugar, Yoghurt, Maida, Ghee, Saffron, Cardaman Powder, Baking Powder', 9: 'Kachori-(195 calories per piece)- Ingredients- Moong Dal, Maida, Ghee, Cumin,Corriander,Mango,Corriander Powder, Fennel, Garam Masala', 10: 'Kofta-(77 calories per serving)- Ingredients- Paneer, Potato, Peas, Milk, Garam Masala, Chilli and Corriander Powder, Corn Starch', 11: 'Ladoo-(204 calories per piece)- Ingredients- Besan, Ghee, Baking Soda, Green Cardamom, Edible Food Color', 12: 'Pani puri-(329 calories per serving)- Ingredients- Puri/Golgappa, Tamarind Chutney, Sev, Corriander/Mint, Potato, Moong, Boondi', 13: 'Paratha-(126 calories per piece)- Ingredients- Potato, Whole wheat, Flour, Ghee, Chilli Pepper, lemon, Cabbbage, Methi', 14: 'Poha-(158 calories per cup)- Ingredients- Poha, lemon, Green chilli, Onion, Potato, Corriander', 15: 'Rasgulla-(120 calories per piece)- Ingredients- Maida, Chhena, Sugar, Milk', 16: 'Samosa-(262 calories per piece)- Ingredients- Potato, Peas, Garam Masala, Wheat flour', 17: 'Vada Pav-(140 calories per piece)- Ingredients- Potato, Green Chilli, Corriander, Garlic, Channa Flour, Bread, Soda'}


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
    
    answer=model1.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    y_new = answer[0][y_class]*100
    predicted_per = y_new[0]

    answer2=model2.predict(img)
    y_class2 = answer2.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class2)
    y = int(y)
    res2 = labels2[y]
    print(res)
    y_new2 = answer2[0][y_class2]*100
    predicted_per2 = y_new2[0]
    
    #answer3=model3.predict(img)
    #y_class3 = answer3.argmax(axis=-1)
    #y = " ".join(str(x) for x in y_class3)
    #y = int(y)
    #res3 = labels3[y]
    #print(res)
    #y_new3 = answer3[0][y_class3]*100
    #predicted_per3 = y_new3[0]
    
    answer4=model4.predict(img)
    y_class4 = answer4.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class4)
    y = int(y)
    res4 = labels4[y]
    print(res)
    y_new4 = answer4[0][y_class4]*100
    predicted_per4 = y_new4[0]

    ret_pred = [res.capitalize(),predicted_per,res2.capitalize(),predicted_per2,res4.capitalize(),predicted_per4]

    return ret_pred

def run():
    ##st.title("Indian Food Recognition")
    st.markdown("""
    <style>
    .big-font {
    font-size:110px !important;
    text-align: center; color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Indian Food Recognition</p>', unsafe_allow_html=True)
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
    
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png","jpeg"])
    if img_file is not None:
        img = Image.open(img_file).resize((250,250))
        st.image(img,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            result= processed_img(save_image_path)
            print(result[0])
            #st.success(" Resnet Predicted : "+result[0]+' with '+ "accuracy: "+str(result[1]))
            #st.success("Mobilenet Predicted : "+result[2]+' with '+ "accuracy: "+str(result[3]))
            #if result[1]> result[3]:
                #st.success(" Resnet Predicted : "+result[0]+' with '+ "accuracy: "+str(result[1]))
            #else:
                #st.success("Mobilenet Predicted : "+result[2]+' with '+ "accuracy: "+str(result[3]))
    
            if result[1]<20 :
                st.success("Low prediction accuracy!!!. Upload a clear Image")
            elif (result[0]=="Dosa-(113 calories per dosa)" and result[1]<20 ):
                st.success("Low prediction accuracy!!!. Upload a clear Image")
            else:
                st.success("Predicted : "+result[0]+' with '+ "accuracy: "+str(result[3]))
                st.success(" Resnet Predicted : "+result[0]+' with '+ "accuracy: "+str(result[1]))
                st.success("Mobilenet Predicted : "+result[2]+' with '+ "accuracy: "+str(result[3]))
                #st.success("VGG Predicted : "+result[4]+' with '+ "accuracy: "+str(result[5]))
                st.success("InceptionNet Predicted : "+result[4]+' with '+ "accuracy: "+str(result[5]))
                

run()