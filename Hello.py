import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Chargement des données (contenant les noms des films, les identifiants de films et les genres)
data = pd.read_csv('df_merge.csv')
df_ratings = pd.read_csv('df_ratings.csv')
df_movies = pd.read_csv('df_movies.csv')

def generate_recommendation(model, user_id, ratings_df, movies_df, n_items):
    # Obtenez une liste de tous les identifiants de films à partir de l'ensemble de données
    movie_ids = df_ratings["movieId"].unique()
    # Obtenir une liste de tous les films qui ont été regardés par l'utilisateur.
    movie_ids_user = df_ratings.loc[df_ratings["userId"] == user_id, "movieId"]
    # Obtenir une liste de tous les films IDS qui n'ont pas été regardés par l'utilisateur
    movie_ids_to_pred = np.setdiff1d(movie_ids, movie_ids_user)
 
   # Appliquer une note de 4 à toutes les interactions (uniquement pour correspondre au format de l'ensemble de données Surprise)
    test_set = [[user_id, movie_id, 4] for movie_id in movie_ids_to_pred]
 
   # Prévoir les évaluations et générer des recommandations
    predictions = model.test(test_set)
    pred_ratings = np.array([pred.est for pred in predictions])
    print("Top {0} recommandations d'éléments pour l'utilisateur {1}:".format(n_items, user_id))
   # Classer les films les plus populaires en fonction des évaluations prédites
    index_max = (-pred_ratings).argsort()[:n_items]
    for i in index_max:
        movie_id = movie_ids_to_pred[i]
        print(df_movies[df_movies["movieId"]==movie_id]["title"].values[0], pred_ratings[i])

# Charger le modèle hybride
hybrid_model = joblib.load("svdpp_model.pkl")

# Interface utilisateur Streamlit
st.title("Système de recommandation de films")

# Formulaire de saisie utilisateur
user_id = st.text_input("Entrez l'ID de l'utilisateur:")
n_items = st.text_input("Entrez l'ID du film:")

if user_id and movie_id:
    try:
        user_id = int(user_id)
        n_items = int(n_items)

        # Faire une recommandation en utilisant le modèle hybride
        prediction = generate_recommendation(hybrid_model,user_id,df_ratings,df_movies,n_items)

        st.write(f"La prédiction de la note pour l'utilisateur {user_id} et le film {n_items} est {prediction:.2f}")
    except ValueError:
        st.write("Veuillez entrer des valeurs numériques valides.")
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
        ### Want to learn more?
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)
        ### See more complex demos
        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )


if __name__ == "__main__":
    run()
