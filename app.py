import streamlit as st

# adiing title of your app

st.title("my first app")

#add simple text

st.text("We create simple text")

#user input
number =st.slider("pick a number", 0,100)

st.write(f"you picked: {number}")

#adding a button

if st.button("say hello"):
    st.write("hello there")
else:
    st.write("goodbye")

#add radio button with options

genre = st.radio(
    "what is your favourite color",
    ("blue","red", "green")
)

st.write(f'you selected: {genre}')

#add a drop down list

option = st.selectbox(
    "what is your favourite fruit", 
    ("mango","banana", "orange"))

st.write(f'Your favourite fruit is: {option}')

#add a drop down list on the left sidebar

option1 = st.sidebar.selectbox(
    'how would you like to be contacted',
    ('email', 'home phone', 'mobile phone')
)

#add text_input

st.text_input('enter your whatsapp number')

#add a file uploader

uploaded_file = st.sidebar.file_uploader('choose a file ', type = "csv")
