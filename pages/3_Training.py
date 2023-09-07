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


st.title("Decision Tree Classification Model")

da = pd.read_csv("train.csv")
da = da.drop(['Ticket'], axis=1)
da = da.drop(['SibSp'], axis=1)
da = da.drop(['Parch'], axis=1)
da = da.drop(['Cabin'], axis=1)
da = da.drop(['Embarked'], axis=1)
da = da.drop(['PassengerId'], axis=1)
da = da.drop(['Name'], axis=1)
da = da.replace(to_replace='None', value=np.nan).dropna()
da = da.replace(to_replace='male', value=0)
da = da.replace(to_replace='female', value=1)
da.Age = da.Age.round()
da.Fare = da.Fare.round(2)


#split our data into input/output
#outputY is the survived column
outputY = da.Survived


#inputX is all the other data
db = da.drop("Survived", axis=1)
inputX = db







st.subheader("\nTraining")
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
