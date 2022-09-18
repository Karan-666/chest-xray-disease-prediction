import streamlit as st
import base64

main_bg = "bg.jpg"
main_bg_ext = "jpg"

side_bg = "bg.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import numpy as np
import skimage
#import sklearn
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image
st.title('X-Ray Covid Predictor:')

model = pickle.load(open('img_model.p', 'rb'))

uploaded_file = st.file_uploader("choose an image...", type=["jpg", "png","jpeg"])
if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img,caption='Uploaded Image')

  if st.button('PREDICT'):
    CATEGORIES = ['Covid','Normal','Viral Pneumonia']
    st.write('Result....')
    flat_data = []
    img = np.array(img)
    img_resized = resize(img,(150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_out = model.predict(flat_data)
    y_out = CATEGORIES[y_out[0]]
    st.title(f'PREDICTED OUTPUT: {y_out}')
    
    q= model.predict_proba(flat_data)
    for index, item in enumerate(CATEGORIES):
        st.write(f'{item} : {q[0][index]*100} %')





        
