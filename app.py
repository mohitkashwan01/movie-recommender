import pandas as pd                                             # for data manipulation and tabular data work
from sklearn.feature_extraction.text import TfidfVectorizer     # converts to numeric vectors (TF IDF)
from sklearn.metrics.pairwise import cosine_similarity          # measure similarity
import streamlit as st
#for website making and UI development of code: streamlit: terminal run: python -m streamlit run test6final.py

#Loading dataset
#read data
df = pd.read_csv(r"netflix_titles.csv")

# Handle missing values
for col in ['director', 'cast', 'description', 'listed_in']: 
    df[col] = df[col].fillna('')    #filling empty values with empty string to avoid errors (null value)

# Combine useful text features
# description + genre + cast + director all together
df['combined'] = (
    df['description'] + " " +
    df['listed_in'] + " " +
    df['cast'] + " " +
    df['director']
)

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')      # converts to vector form, ignore words like and.....
tfidf_matrix = tfidf.fit_transform(df['combined'])  #Converts the combined text column into a matrix of numeric vectors


# Cosine similarity as high dimension data 
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)  #2d matrix

# Title indices
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

# Recommendation function
# accepting user input
def recommend_multiple(titles, cosine_sim=cosine_sim):
    titles = [t.lower() for t in titles]      #lower case conversion of all values
    indices_list = [indices[t] for t in titles if t in indices]    #finding in dataset


# no match found
    if not indices_list:
        return pd.DataFrame([{
            "title": "No matching titles found",
            "type": "-",
            "release_year": "-",
            "listed_in": "-",
            "description": "-"
        }])

# average similarity score
    sim_scores = sum([cosine_sim[idx] for idx in indices_list]) / len(indices_list)
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    movie_indices = [i[0] for i in sim_scores if df['title'].iloc[i[0]].lower() not in titles][:20] # top 20 recommendations
    return df.iloc[movie_indices][['title', 'type', 'release_year', 'listed_in', 'description']]

# Streamlit UI
st.set_page_config(page_title="Netflix Recommender", layout="wide")

# defining the look and design 
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@300;600&display=swap');
    body {
        background-color: #1E1B18;
        color: #E5D4B0;
    }
    .stButton>button {
        background-color: #8B0000;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: 2px solid transparent;
    }
    .netflix-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 46px;
        color: #E50914; /* Netflix Red */
        letter-spacing: 2px;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }
    .stButton>button:hover {
        background-color: #A52A2A;
        color: #FFFACD;
        border: 2px solid #E50914; /* Netflix red accent border */
    }
    .title-accent {
        border-left: 6px solid #E50914; /* Netflix red bar */
        padding-left: 10px;
        font-size: 28px;
        font-weight: bold;
    }
    .expander-description {
        border-top: 2px solid #E50914; /* subtle red line inside expanders */
        margin-top: 5px;
        padding-top: 5px;
        color: #E5D4B0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="netflix-title">🎬🍿 Netflix Recommender System</div>', unsafe_allow_html=True)

#st.markdown('<div class="title-accent">🎬🍿Netflix Recommender System </div>', unsafe_allow_html=True)

st.write("***Discover your next favorite film with ease—our movie recommender app curates personalized picks based on your unique taste and mood. Whether you're craving a thrilling adventure or a quiet indie gem, let us guide your cinematic journey with elegance and precision.***")

# Input box
user_input = st.text_input("Enter movie/show names:", "")

if st.button("Get Recommendations"):
    if not user_input.strip():
        st.warning("Please enter at least one movie/show name!")
    else:
        user_titles = [t.strip() for t in user_input.split(",")]
        recommendations = recommend_multiple(user_titles)

        st.subheader("Recommended for You ✨")
        #st is streamlit  for adding subheader and input box
        # Display recommendations as expandable list with styled description
        for _, row in recommendations.iterrows():    # loop for each recommended movie
            with st.expander(f"{row['title']} ({row['release_year']}) | {row['type']} | {row['listed_in']}"):   #expand box for each movie
                #displays the description of recommended movied with format 
                st.markdown(f"<div class='expander-description'>{row['description']}</div>", unsafe_allow_html=True)