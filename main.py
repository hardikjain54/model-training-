import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# add title
st.title('Data Analysis App')
st.subheader('this is a simple data analysis application created by @hardik')

#create a dropdown list to choose a dataset
dataset_options = ['iris', 'titanic','tips','diamonds']
selected_dataset = st.selectbox('select a dataset', dataset_options)

#load the selected dataset
if selected_dataset == 'iris':
    df = sns.load_dataset('iris')
elif selected_dataset == 'titanic':
    df = sns.load_dataset('titanic')
elif selected_dataset == 'tips':
    df = sns.load_dataset('tips')
elif selected_dataset == 'diamonds':
    df = sns.load_dataset('diamonds')

#button to upload custom dataset

uploaded_file = st.file_uploader('upload a custom dataset', type=[ "csv","xlsx"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        df = pd.read_excel(uploaded_file)
                   
#display the dataset
st.write(df.head())
st.write('shape of the dataset:', df.shape)

#display column names of selected data with their data types
st.write("Column names and their data types:" , df.dtypes)
#print null values
if df.isnull().sum().sum()>0:
    st.write("Null values in the dataset:",df.isnull().sum().sort_values(ascending=False))
else:
    st.write("No null values in the dataset")

#display summary statistics
st.write("Summary statistics of the dataset:", df.describe())

#select the specific column in the dataset to plot the graph

x_axis = st.selectbox('select x-axis', df.columns)
y_axis = st.selectbox('select y-axis', df.columns)

plot_type = st.selectbox("select plot type",['line','scatter','bar','histogram','pie chart','boxplot'])

#plot the data
if plot_type == 'line':
    st.line_chart(df[[x_axis,y_axis]]) 
elif plot_type == 'scatter':
    st.write(sns.scatterplot(data=df, x=x_axis, y=y_axis))
    st.pyplot()
elif plot_type == 'bar':
    st.bar_chart(df[[x_axis,y_axis]])
elif plot_type == 'histogram':
    st.write(sns.histplot(data=df, x=x_axis, kde=True))
    st.pyplot()
elif plot_type == 'pie chart':
    pie_data = df[x_axis].value_counts()
    st.write(pie_data)
    st.write(plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%'))
    st.pyplot()
elif plot_type == 'boxplot':
    st.write(sns.boxplot(data=df, x=x_axis, y=y_axis))
    st.pyplot() 

#create a pair plot

st.subheader('Pair Plot')
#select the column to be used as hue in pairplot

hue_column = st.selectbox("select a column to be used as hue column",df.columns)
st.pyplot(sns.pairplot(df,hue = hue_column))