import pandas as pd
from flask import Flask, render_template,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("C:\\Users\\gan8k\\OneDrive - Contoso\\Documents\\Kishore - Personal\\Data Science\\Recommendation Engine Project\\Gitb\\Revised Dataset - Restaurant.csv")


# In[4]:


# Getting the file information
df.info()

# Finding the null value in the column
df.isna().sum()


# In[5]:


# Importing the numpy library for numeric calculation
import numpy as np

# Importing the Simple Imputer to get the null value
from sklearn.impute import SimpleImputer


# In[6]:


# Mean Imputer for Numerical Data 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df["Rating"] = pd.DataFrame(mean_imputer.fit_transform(df[["Rating"]]))
df["Rating"].isna().sum()


# In[7]:


# Changing the numerical value to categorical value using Discretization

object = pd.cut(df.Age,bins = [0,16,30,38,45,60,99],labels=['Child','Young Aged Adults', 'Middle Aged Adults', 'Old Aged Adults','Old Aged','Elderly'])
object1 = pd.cut(df.TotalBill,bins = [0,1000,1500,2000],labels=['Low Fair','Medium Fair', 'High Fair'])
object2 = pd.cut(df.Rating,bins = [0,1,2,3,4,5],labels=['Worst','Poor', 'Average', 'Good','Excellent'])


# In[8]:


# Inserting a new column for the categorical value

df.insert(4,'AgeGroup', object)
df.insert(22,'BillGroup', object1)
df.insert(24,'RatingGroup', object2)


# In[9]:


#Checking the datatypes of the column value
df.dtypes


# In[10]:


# Changing the category data types to object
df = df.astype({"AgeGroup":'object',"BillGroup":'object',"RatingGroup":'object' })


# In[11]:


#Checking the datatypes of the column value
df.dtypes


# In[12]:


# Changing the Dataframe name to anime
anime = df

# Checking the shape (showing the number of rows & column)
anime.shape 

# Checking the column name (showing the column name)
anime.columns


# In[13]:


#term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
from sklearn.feature_extraction.text import TfidfVectorizer 


# In[14]:


# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 


# In[15]:


# Preparing the Tfidf matrix by fitting and transforming
#Initiated all possible criteria based on Customer Details which be using to arrive the results.

tfidf_matrix = tfidf.fit_transform(anime['DiningType']+anime['Variety']+anime['AgeGroup']+anime['EmailID']+anime['Day']+anime['PreferrredIngredients']+anime['MainIngredients1']+anime['BillGroup']+anime['RatingGroup'])
tfidf_matrix1 = tfidf.fit_transform(anime['DiningType']+anime['Variety']+anime['AgeGroup']+anime['EmailID']+anime['Day']+anime['PreferrredIngredients']+anime['MainIngredients2']+anime['BillGroup']+anime['RatingGroup'])
tfidf_matrix2 = tfidf.fit_transform(anime['DiningType']+anime['Variety']+anime['AgeGroup']+anime['EmailID']+anime['Day']+anime['PreferrredIngredients']+anime['MainIngredients2']+anime['BillGroup']+anime['RatingGroup'])
tfidf_matrix.shape 


# In[16]:


#importing the linear kernel library
from sklearn.metrics.pairwise import linear_kernel


# In[17]:


# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim_matrix1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)
cosine_sim_matrix2 = linear_kernel(tfidf_matrix2, tfidf_matrix2)


# In[18]:


# creating a mapping of anime name to index number 
anime_index = pd.Series(anime.index, index = anime['Order1']).drop_duplicates()
anime_index1 = pd.Series(anime.index, index = anime['Order2']).drop_duplicates()
anime_index2 = pd.Series(anime.index, index = anime['Order3']).drop_duplicates()


def get_recommendations(Order1, cosine_sim):
    global result
 cosine_scores = list(enumerate(cosine_sim_matrix[anime_id]))
 cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
 cosine_scores_N = cosine_scores[0: topN+1]
 anime_idx  =  [i[0] for i in cosine_scores_N]
 anime_scores =  [i[1] for i in cosine_scores_N]
 result = pd.DataFrame(columns=["Order1", "Score"])
 result["Order1"] = anime.loc[anime_idx, "Order1"]
 result["Score"] = anime_scores
 result.reset_index(inplace = True)  
 # anime_similar_show.drop(["index"], axis=1, inplace=True)
 print (result)
 return (result)

get_recommendations("Idli", topN = 10)

# If we use the same order on order2 & Order3, the recommendation will be different, will not provide the same output.
# IF we recommend the same item it would not be correct, hence the suggestion will be different based on the order items.

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/about',methods=['POST'])
def getvalue():
    Ordername = request.form['Ordername']
    get_recommendations(Ordername,topN)
    dfs=result
    return render_template('result.html',  tables=[dfs.to_html(classes='data')], titles=dfs.columns.values)

if __name__ == '__main__':
    app.run(debug=False)
