import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the MovieLens dataset
url_data = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
url_item = "http://files.grouplens.org/datasets/movielens/ml-100k/u.item"

column_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv(url_data, sep="\t", names=column_names)

# Load movie titles
movie_columns = [
    "item_id",
    "title",
    "release_date",
    "video_release_date",
    "IMDb_URL",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
df_movies = pd.read_csv(url_item, sep="|", names=movie_columns, encoding="latin-1")

# Create a mapping of item_id to movie title
movie_titles = dict(zip(df_movies["item_id"], df_movies["title"]))

# Convert the dataframe to Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

# Split the data into training and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Use SVD (Singular Value Decomposition) algorithm
model = SVD()

# Train the model on the training set
model.fit(trainset)

# Predict ratings for the test set
predictions = model.test(testset)

# Compute and print the accuracy
accuracy.rmse(predictions)


# Function to recommend top N items for a given user
def recommend(user_id, num_recommendations=5):
    # Get a list of all item_ids
    all_items = df["item_id"].unique()

    # Predict ratings for all items
    predicted_ratings = [model.predict(user_id, item_id).est for item_id in all_items]

    # Create a list of item_id and their predicted ratings
    item_ratings = list(zip(all_items, predicted_ratings))

    # Sort the items by predicted ratings in descending order
    item_ratings.sort(key=lambda x: x[1], reverse=True)

    # Get the top N items
    top_items = item_ratings[:num_recommendations]

    # Convert item_ids to movie titles
    top_items_with_titles = [
        (movie_titles[item_id], rating) for item_id, rating in top_items
    ]

    # Return the top N recommended items with titles
    return top_items_with_titles


# Example usage: Recommend top 5 items for user with user_id 196
user_id = 196
recommendations = recommend(user_id, 5)
print("Top 5 recommendations for user {}:".format(user_id))
for title, rating in recommendations:
    print(f"{title}: Predicted Rating {rating:.2f}")
