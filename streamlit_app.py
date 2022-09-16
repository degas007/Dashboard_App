#import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title='My app page', page_icon=':relaxed:', layout='wide')

with st.container():
    st.subheader("This is a Dashboard App, Let's have fun together")
    st.title("Let's create the App")
    st.write("App created using Python with open source library streamlit")

#open and display image
image = Image.open('machine learning img.jpg')
st.image(image, caption='Interactive front-end', use_column_width=True)

selected_item = st.radio('Please select to get into page:', ['Breast-cancer prediction', 'Chart', 'ABC', 'Buttons'])
#Title and sub-title
st.write('Diabetes Detection: ')
st.write('Prediction of ML')

with st.container():
    st.write('---')
    left_col, right_col = st.columns(2)
    with left_col:
        st.header('Hello how are you doing?')
        st.write('##')
        st.write('How to customize your page')
    with right_col:
        st.header('Practise brings perfection')
        st.write('##')
        st.write('Motivation is the most important')


if selected_item == 'Breast-cancer prediction':
    #getting the data
    df = pd.read_csv('breast-cancer.csv')
    st.subheader('Information on dataset')
    #dispay the dataset
    st.dataframe(df)

    st.write(df.describe())

    #train test split
    X = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    #split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state=14)

    #train the model
    RFC = RandomForestClassifier()
    RFC.fit(X_train, y_train)
    Score = (str(accuracy_score(y_test, RFC.predict(X_test))*100)+ ' %')

    def get_user_input():
         for i in df.columns[2:]:
            details = (str(i), df[str(i)].min(), df[str(i)].max(), df[str(i)].mean())
            print(i, '=st.sidebar.slider', details)
         for i in df.columns[2:]:
             print(str(i), ':', i)
         features = pd.DataFrame(User_input, index=[0])
         return features


    radius_mean =st.sidebar.slider ('radius_mean', 6.981, 28.11, 14.127291739894552)
    texture_mean =st.sidebar.slider ('texture_mean', 9.71, 39.28, 19.289648506151142)
    perimeter_mean =st.sidebar.slider ('perimeter_mean', 43.79, 188.5, 91.96903339191564)
    area_mean =st.sidebar.slider ('area_mean', 143.5, 2501.0, 654.8891036906855)
    smoothness_mean =st.sidebar.slider ('smoothness_mean', 0.05263, 0.1634, 0.0963602811950791)
    compactness_mean =st.sidebar.slider ('compactness_mean', 0.01938, 0.3454, 0.10434098418277679)
    concavity_mean =st.sidebar.slider ('concavity_mean', 0.0, 0.4268, 0.0887993158172232)
    concave_points_mean =st.sidebar.slider ('concave points_mean', 0.0, 0.2012, 0.04891914586994728)
    symmetry_mean =st.sidebar.slider ('symmetry_mean', 0.106, 0.304, 0.18116186291739894)
    fractal_dimension_mean =st.sidebar.slider ('fractal_dimension_mean', 0.04996, 0.09744, 0.06279760984182776)
    radius_se =st.sidebar.slider ('radius_se', 0.1115, 2.873, 0.40517205623901575)
    texture_se =st.sidebar.slider ('texture_se', 0.3602, 4.885, 1.2168534270650264)
    perimeter_se =st.sidebar.slider ('perimeter_se', 0.757, 21.98, 2.8660592267135327)
    area_se =st.sidebar.slider ('area_se', 6.802, 542.2, 40.337079086116)
    smoothness_se =st.sidebar.slider ('smoothness_se', 0.001713, 0.03113, 0.007040978910369069)
    compactness_se =st.sidebar.slider ('compactness_se', 0.002252, 0.1354, 0.025478138840070295)
    concavity_se =st.sidebar.slider ('concavity_se', 0.0, 0.396, 0.03189371634446397)
    concave_points_se =st.sidebar.slider ('concave points_se', 0.0, 0.05279, 0.011796137082601054)
    symmetry_se =st.sidebar.slider ('symmetry_se', 0.007882, 0.07895, 0.02054229876977153)
    fractal_dimension_se =st.sidebar.slider ('fractal_dimension_se', 0.0008948, 0.02984, 0.0037949038664323374)
    radius_worst =st.sidebar.slider ('radius_worst', 7.93, 36.04, 16.269189806678387)
    texture_worst =st.sidebar.slider ('texture_worst', 12.02, 49.54, 25.677223198594024)
    perimeter_worst =st.sidebar.slider ('perimeter_worst', 50.41, 251.2, 107.26121265377857)
    area_worst =st.sidebar.slider ('area_worst', 185.2, 4254.0, 880.5831282952548)
    smoothness_worst =st.sidebar.slider ('smoothness_worst', 0.07117, 0.2226, 0.13236859402460457)
    compactness_worst =st.sidebar.slider ('compactness_worst', 0.02729, 1.058, 0.25426504393673116)
    concavity_worst =st.sidebar.slider ('concavity_worst', 0.0, 1.252, 0.27218848330404216)
    concave_points_worst =st.sidebar.slider ('concave_points_worst', 0.0, 0.291, 0.11460622319859401)
    symmetry_worst =st.sidebar.slider ('symmetry_worst', 0.1565, 0.6638, 0.2900755711775044)
    fractal_dimension_worst =st.sidebar.slider ('fractal_dimension_worst', 0.05504, 0.2075, 0.0839458172231986)



    User_input = {radius_mean : radius_mean,
    texture_mean : texture_mean,
    perimeter_mean : perimeter_mean,
    area_mean : area_mean,
    smoothness_mean : smoothness_mean,
    compactness_mean : compactness_mean,
    concavity_mean : concavity_mean,
    concave_points_mean : concave_points_mean,
    symmetry_mean : symmetry_mean,
    fractal_dimension_mean : fractal_dimension_mean,
    radius_se : radius_se,
    texture_se : texture_se,
    perimeter_se : perimeter_se,
    area_se : area_se,
    smoothness_se : smoothness_se,
    compactness_se : compactness_se,
    concavity_se : concavity_se,
    concave_points_se : concave_points_se,
    symmetry_se : symmetry_se,
    fractal_dimension_se : fractal_dimension_se,
    radius_worst : radius_worst,
    texture_worst : texture_worst,
    perimeter_worst : perimeter_worst,
    area_worst : area_worst,
    smoothness_worst : smoothness_worst,
    compactness_worst : compactness_worst,
    concavity_worst : concavity_worst,
    concave_points_worst : concave_points_worst,
    symmetry_worst : symmetry_worst,
    fractal_dimension_worst : fractal_dimension_worst
    }
    #transfering user input to dataframe and return
    user_input = get_user_input()

    st.subheader('User_input')
    st.write(user_input)

    #display model accuracy
    st.subheader('Model Accuracy Score:')
    st.write(Score)

    #prediction
    prediction=RFC.predict(user_input)
    list_class = ['Malignant', 'Benign']
    st.subheader('Classification_result: ')
    result = list_class[int(prediction)]
    st.write(result)
    
elif selected_item == 'Chart':
    df_charts = pd.read_csv('Employee_Salary_Dataset.csv')
    st.table(df_charts)
    plt.scatter(df_charts['Age'], df_charts['Salary'])
    plt.title('Salary with respect to Age')
    st.pyplot()
    st.write('.....')
    plt.scatter(df_charts['Experience_Years'], df_charts['Salary'])
    plt.title('Salary with respect to Experience_Years')
    st.pyplot()
    
elif selected_item == 'Buttons':
    st.subheader('Still working with it, any new idea???')
    st.sidebar.slider('abc, 1, 10, 5')
    st.sidebar.number_input('Pick a number', 0, 10)
    st.sidebar.multiselect('Technology :', ['Python', 'web-scrapping', 'Text analysis', 'AWS'])
    st.sidebar.selectbox('Gender', ['Male', 'Female'])
    
    image = Image.open('well done.jpg')
    if st.sidebar.button('click me'):
        on_click = st.image(image)
    if st.sidebar.button('emotion'):
        on_click = st.sidebar.write(':blush:')

elif selected_item == 'ABC':
    st.title("You've selected ABC: Here you go A B C")
    st.subheader('**** A B C ****')

    import webbrowser 

    Cheatsheet = 'https://docs.streamlit.io/library/cheatsheet'
    Tutorial = 'https://docs.streamlit.io/knowledge-base/tutorials'
    Deploying_Heroku = 'https://www.youtube.com/watch?v=nJHrSvYxzjE&ab_channel=CodingIsFun'
    Deploying_AWS_S3bucket= 'https://docs.streamlit.io/knowledge-base/tutorials/databases/aws-s3'
    Video_tutorial_series = 'https://www.youtube.com/watch?v=FOULV9Xij_8&list=PL7QI8ORyVSCaejt2LICRQtOTwmPiwKO2n&ab_channel=CodingIsFun'
    Video_tutorial_series_1 = 'https://www.youtube.com/watch?v=UN4DaSAZel4&list=PLuU3eVwK0I9PT48ZBYAHdKPFazhXg76h5&ab_channel=HarshGupta'

    if st.button('Cheatsheet link'):
        webbrowser.open_new_tab(Cheatsheet)
    elif st.button('Tutorial_Link'):
        webbrowser.open_new_tab(Tutorial)
    elif st.button('Deploying with Heroku'):
        webbrowser.open_new_tab(Deploying_Heroku)
    elif st.button('Deploying with AWS s3 bucket'):
        webbrowser.open_new_tab(Deploying_AWS_S3bucket)
    elif st.button('Video_tutorial_series'):
        webbrowser.open_new_tab(Video_tutorial_series)
    elif st.button('Video_tutorial_series_1'):
        webbrowser.open_new_tab(Video_tutorial_series_1)