import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.metrics  import f1_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import tree


st.title("Titanic Survival Rates: In the Data")
st.subheader("Data Cleaning and Visualizations")
st.write("Here is the raw data set from Kaggle. It contains tons of information about \
           the passengers; however, not all of this data is required. Lets clean the data \
           by removing some columns to help us identify trends.")

da = pd.read_csv("train.csv")
st.dataframe(da, hide_index=True)

st.write("To clean, lets remove the ticket numbers, id, names, cabin, port of embarkation, \
             number of siblings, and number of children/parents.")

da = da.drop(['Ticket'], axis=1)
da = da.drop(['SibSp'], axis=1)
da = da.drop(['Parch'], axis=1)
da = da.drop(['Cabin'], axis=1)
da = da.drop(['Embarked'], axis=1)
da = da.drop(['PassengerId'], axis=1)
da = da.drop(['Name'], axis=1)

st.dataframe(da, hide_index=True)

st.write("Much better! More cleaning is required though, as some age values say 'None'\
           Lets remove all values with an invalid age, then round the age and fare! \
           Additionally, in order to prep the data for the model, lets assign '1' to mean 'female', \
           and '0' to mean 'male'.")

da = da.replace(to_replace='None', value=np.nan).dropna()
da = da.replace(to_replace='male', value=0)
da = da.replace(to_replace='female', value=1)
da.Age = da.Age.round()
da.Fare = da.Fare.round(2)
st.dataframe(da, hide_index=True)

st.write("Perfect. Now lets prepare the data to be put through a machine learning model using classification \
           by splitting up the data into inputs and outputs. The output will be the survived column, and the input will \
           be everything else.")

#split our data into input/output
#outputY is the survived column
outputY = da.Survived
st.dataframe(outputY, hide_index=True)

#inputX is all the other data
db = da.drop("Survived", axis=1)
inputX = db

st.dataframe(db, hide_index=True)





st.subheader("\nMachine Learning Model Training/Testing")
st.write("We will be using a decision tree classifier, which uses a decision tree to \
           classify whether or not a passenger survived")


slida = st.slider("Choose a max depth of neural network:", value=10, max_value=50, min_value=1)

def trainJ(depth):
    X_train, X_test, y_train, y_test = train_test_split(inputX, outputY, test_size=0.25)
    DT2 = DecisionTreeClassifier(max_depth=depth)
    DT2.fit(X_train,y_train)
    fig = plt.figure(figsize=(30,30))
    prediction = DT2.predict(X_test)
    tree.plot_tree(DT2, filled=True, feature_names=["Class", "Sex", "Age", "Fare"], class_names=["Lived", "Died"])
    fig.savefig("decision_tree.png")
    st.write("The accuracy score of the model is: " + str(np.round(accuracy_score(y_test,prediction), 6)))
    st.write("The mean absolute error of the model is: " + str(np.round(mean_absolute_error(y_test,prediction), 6)))
    st.image("decision_tree.png")

st.button("Click to train model!", on_click=trainJ(slida))













#fig = px.pie(
#    da,
#    values='Survived',
#    names='Sex',
#    title='Percentage of total survived by Sex'
#)
#
#fig2 = px.histogram(da, x="Sex", y="Survived",
#             color='Sex', barmode='group',
#             height=400)
#
#st.plotly_chart(fig)
#st.plotly_chart(fig2)
