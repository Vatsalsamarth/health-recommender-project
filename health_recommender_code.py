# Full Python Code for Personalized Health Monitoring & Recommender System

# --- 1. SETUP AND LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import os

# Import tensorflow for the neural network model
# Ensure you have tensorflow installed: pip install tensorflow
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

print("Libraries imported successfully.")

# Create a directory to save plots
if not os.path.exists('plots'):
    os.makedirs('plots')
    
# --- 2. DATA SIMULATION ---
# We simulate data for a more comprehensive and reproducible example.
np.random.seed(42)
num_users = 5000
num_tips = 50

# Create user profiles
users_df = pd.DataFrame({
    'user_id': range(num_users),
    'age': np.random.randint(18, 65, size=num_users),
    'gender': np.random.choice(['Male', 'Female'], size=num_users, p=[0.5, 0.5]),
    'avg_steps': np.random.normal(8000, 2500, size=num_users).astype(int),
    'avg_hr': np.random.normal(75, 10, size=num_users).astype(int),
    'avg_sleep': np.random.normal(7, 1.5, size=num_users).round(1)
})
users_df['avg_steps'] = users_df['avg_steps'].clip(1000, 20000)
users_df['avg_sleep'] = users_df['avg_sleep'].clip(4, 10)

# Create health tips
tip_categories = ['Cardio', 'Strength', 'Mindfulness', 'Nutrition', 'Sleep Hygiene']
tips_df = pd.DataFrame({
    'tip_id': range(num_tips),
    'category': np.random.choice(tip_categories, size=num_tips),
    'tip_text': [f"Health Tip #{i+1}" for i in range(num_tips)]
})

# Create user-tip interactions (ratings)
# This simulates users adopting or liking certain tips
ratings_list = []
for user_id in users_df['user_id']:
    # Each user rates a random number of tips
    num_ratings = np.random.randint(5, 20)
    rated_tips = np.random.choice(tips_df['tip_id'], size=num_ratings, replace=False)
    for tip_id in rated_tips:
        # Rating is influenced by the match between user stats and tip category
        user_profile = users_df.loc[user_id]
        tip_category = tips_df.loc[tip_id]['category']
        
        base_rating = np.random.randint(1, 6) # 1 to 5 stars
        
        # Add some logic to make ratings non-random
        if tip_category == 'Cardio' and user_profile['avg_steps'] < 6000:
            base_rating = np.clip(base_rating + np.random.choice([1, 2]), 1, 5)
        if tip_category == 'Sleep Hygiene' and user_profile['avg_sleep'] < 6.5:
            base_rating = np.clip(base_rating + np.random.choice([1, 2]), 1, 5)
        if tip_category == 'Mindfulness' and user_profile['avg_hr'] > 80:
             base_rating = np.clip(base_rating + np.random.choice([1, 2]), 1, 5)
            
        ratings_list.append({'user_id': user_id, 'tip_id': tip_id, 'rating': base_rating})

ratings_df = pd.DataFrame(ratings_list)

print("Data simulation complete.")
print(f"Generated {len(users_df)} users, {len(tips_df)} tips, and {len(ratings_df)} interactions.")


# --- 3. EXPLORATORY DATA ANALYSIS (EDA) ---
print("\nStarting Exploratory Data Analysis...")
plt.style.use('seaborn-v0_8-whitegrid')

# Histogram: Steps per day
plt.figure(figsize=(10, 6))
sns.histplot(users_df['avg_steps'], bins=30, kde=True)
plt.title('Distribution of Average Daily Steps per User', fontsize=16)
plt.xlabel('Average Steps', fontsize=12)
plt.ylabel('Number of Users', fontsize=12)
plt.axvline(users_df['avg_steps'].mean(), color='r', linestyle='--', label=f"Mean: {users_df['avg_steps'].mean():.0f}")
plt.legend()
plt.savefig('plots/eda_steps_histogram.png')
plt.close()
print("Saved: eda_steps_histogram.png")

# Time Series (Simulated): Average heart rate vs. hour
# For this, we'll generate some plausible hourly data for a sample user
hours = np.arange(24)
hr_day = 70 + 15 * np.sin((hours - 8) * np.pi / 12) + np.random.normal(0, 3, 24)
hr_day[13] += 10 # Lunch walk
plt.figure(figsize=(12, 6))
plt.plot(hours, hr_day, marker='o', linestyle='-')
plt.title('Simulated Heart Rate Over a 24-Hour Period', fontsize=16)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Heart Rate (BPM)', fontsize=12)
plt.xticks(np.arange(0, 25, 2))
plt.grid(True)
plt.savefig('plots/eda_hr_timeseries.png')
plt.close()
print("Saved: eda_hr_timeseries.png")

# Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = users_df[['age', 'avg_steps', 'avg_hr', 'avg_sleep']].corr()
sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f')
plt.title('Correlation Heatmap of User Health Metrics', fontsize=16)
plt.savefig('plots/eda_correlation_heatmap.png')
plt.close()
print("Saved: eda_correlation_heatmap.png")

# Boxplot: Sleep duration by gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='avg_sleep', data=users_df)
plt.title('Distribution of Sleep Duration by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Average Sleep (Hours)', fontsize=12)
plt.savefig('plots/eda_sleep_boxplot.png')
plt.close()
print("Saved: eda_sleep_boxplot.png")

print("EDA complete. Plots saved in 'plots/' directory.")

# --- 4. CONTENT-BASED RECOMMENDER: USER PROFILE CLUSTERING ---
print("\nBuilding Content-Based Recommender (User Clustering)...")
features = users_df[['age', 'avg_steps', 'avg_hr', 'avg_sleep']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
users_df['cluster'] = kmeans.fit_predict(scaled_features)

# PCA for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

# Plotting the clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=users_df['cluster'], cmap='viridis', alpha=0.7)
plt.title('User Clusters based on Health Metrics (PCA-reduced)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)

# Annotate cluster centers
centers = pca.transform(kmeans.cluster_centers_)
cluster_names = {
    0: "Active Sleepers", 1: "Sedentary Night-Owls",
    2: "Moderate & Steady", 3: "Active but Sleep-Deprived"
}
for i, center in enumerate(centers):
    plt.scatter(center[0], center[1], s=250, c='red', marker='X')
    plt.text(center[0], center[1]+0.2, cluster_names.get(i, f'Cluster {i}'), fontsize=12, ha='center', color='black')

plt.legend(handles=scatter.legend_elements()[0], labels=cluster_names.values())
plt.savefig('plots/model_user_clusters.png')
plt.close()
print("User clustering complete. Plot saved.")


# --- 5. COLLABORATIVE FILTERING MODELS ---
print("\nBuilding Collaborative Filtering Models...")

# Prepare data for CF models
X = ratings_df[['user_id', 'tip_id']].values
y = ratings_df['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_df = pd.DataFrame(X_train, columns=['user_id', 'tip_id'])
train_df['rating'] = y_train
test_df = pd.DataFrame(X_test, columns=['user_id', 'tip_id'])
test_df['rating'] = y_test

# Create user-item matrix for training
user_item_matrix = train_df.pivot_table(index='user_id', columns='tip_id', values='rating').fillna(0)

# --- 5a. KNN-Based Collaborative Filtering ---
print("Training KNN Model...")
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(user_item_matrix)

# KNN Prediction (example for one user)
def get_knn_recs(user_id, k=5):
    if user_id not in user_item_matrix.index:
        return []
    distances, indices = knn.kneighbors(user_item_matrix.loc[user_id, :].values.reshape(1, -1), n_neighbors=k+1)
    # Get neighbors' ratings and average them
    neighbor_ratings = user_item_matrix.iloc[indices.flatten()[1:]]
    avg_ratings = neighbor_ratings.mean(axis=0)
    # Filter out tips the user already rated
    user_rated_tips = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
    avg_ratings = avg_ratings.drop(user_rated_tips, errors='ignore')
    return avg_ratings.nlargest(5).index.tolist()

# --- 5b. NMF-Based Collaborative Filtering ---
print("Training NMF Model...")
nmf = NMF(n_components=20, init='random', random_state=42, max_iter=500)
W = nmf.fit_transform(user_item_matrix) # User features
H = nmf.components_ # Item features
reconstructed_matrix = np.dot(W, H)

# --- 5c. Neural Network Embedding Collaborative Filtering ---
print("Training Neural Network Model...")
n_users = ratings_df['user_id'].nunique()
n_tips = ratings_df['tip_id'].nunique()
embedding_size = 32

# Model architecture
user_input = Input(shape=[1], name='UserInput')
user_embedding = Embedding(n_users, embedding_size, name='UserEmbedding')(user_input)
user_vec = Flatten(name='FlattenUser')(user_embedding)

tip_input = Input(shape=[1], name='TipInput')
tip_embedding = Embedding(n_tips, embedding_size, name='TipEmbedding')(tip_input)
tip_vec = Flatten(name='FlattenTip')(tip_embedding)

dot_product = Dot(axes=1, name='DotProduct')([user_vec, tip_vec])
model_nn = Model(inputs=[user_input, tip_input], outputs=dot_product)
model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model_nn.fit(
    [train_df.user_id, train_df.tip_id], train_df.rating,
    epochs=10,
    batch_size=128,
    verbose=0, # Set to 1 to see training progress
    validation_data=([test_df.user_id, test_df.tip_id], test_df.rating)
)
print("NN Model training complete.")


# --- 6. MODEL EVALUATION ---
print("\nEvaluating Collaborative Filtering Models...")
results = {}

# Make predictions on the test set
test_users = test_df['user_id'].values
test_tips = test_df['tip_id'].values
true_ratings = test_df['rating'].values

# --- 6a. KNN Evaluation ---
# KNN is harder to evaluate with RMSE directly, so we focus on Precision@k
# We'll use a simplified prediction for RMSE: average of neighbors' ratings for a given item
knn_preds = []
for u, i in zip(test_users, test_tips):
    if u in user_item_matrix.index:
        distances, indices = knn.kneighbors(user_item_matrix.loc[u, :].values.reshape(1, -1), n_neighbors=11)
        neighbor_ids = user_item_matrix.index[indices.flatten()[1:]]
        # Get ratings from neighbors for the specific item `i`
        neighbor_ratings_for_item = user_item_matrix.loc[neighbor_ids, i]
        pred = neighbor_ratings_for_item[neighbor_ratings_for_item > 0].mean()
        knn_preds.append(pred)
    else: # Cold start user
        knn_preds.append(train_df['rating'].mean()) # Predict global average
knn_preds = pd.Series(knn_preds).fillna(train_df['rating'].mean()).values
results['KNN'] = {'RMSE': np.sqrt(mean_squared_error(true_ratings, knn_preds))}

# --- 6b. NMF Evaluation ---
nmf_preds = []
for u, i in zip(test_users, test_tips):
    if u in user_item_matrix.index and i in user_item_matrix.columns:
        pred = reconstructed_matrix[user_item_matrix.index.get_loc(u), user_item_matrix.columns.get_loc(i)]
        nmf_preds.append(pred)
    else: # Cold start
        nmf_preds.append(train_df['rating'].mean())
results['NMF'] = {'RMSE': np.sqrt(mean_squared_error(true_ratings, nmf_preds))}

# --- 6c. NN Evaluation ---
nn_preds = model_nn.predict([test_users, test_tips]).flatten()
results['NN Embedding'] = {'RMSE': np.sqrt(mean_squared_error(true_ratings, nn_preds))}

# --- 6d. Precision@5 Calculation ---
# For each user in the test set, get top 5 recommendations and see if any match their highly-rated items
def calculate_precision_at_k(model_name, k=5):
    hits = 0
    total_users = 0
    
    # Get ground truth: items rated highly (4 or 5) by users in the test set
    test_user_positives = test_df[test_df.rating >= 4].groupby('user_id')['tip_id'].apply(list).to_dict()

    for user_id, true_positives in test_user_positives.items():
        if user_id not in user_item_matrix.index: continue
        total_users += 1
        
        # Get top-k recommendations
        if model_name == 'KNN':
            recs = get_knn_recs(user_id, k)
        elif model_name == 'NMF':
            user_idx = user_item_matrix.index.get_loc(user_id)
            scores = np.dot(W[user_idx], H)
            user_rated_tips = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
            scores = pd.Series(scores, index=user_item_matrix.columns).drop(user_rated_tips, errors='ignore')
            recs = scores.nlargest(k).index.tolist()
        elif model_name == 'NN Embedding':
            user_input_arr = np.array([user_id] * n_tips)
            tip_input_arr = np.arange(n_tips)
            scores = model_nn.predict([user_input_arr, tip_input_arr]).flatten()
            user_rated_tips = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
            scores = pd.Series(scores, index=range(n_tips)).drop(user_rated_tips, errors='ignore')
            recs = scores.nlargest(k).index.tolist()
        else:
            recs = []
            
        # Check for hits
        hit_count = len(set(recs) & set(true_positives))
        if hit_count > 0:
            hits += 1
            
    return hits / total_users if total_users > 0 else 0

results['KNN']['Precision@5'] = calculate_precision_at_k('KNN')
results['NMF']['Precision@5'] = calculate_precision_at_k('NMF')
results['NN Embedding']['Precision@5'] = calculate_precision_at_k('NN Embedding')

# Create and display the evaluation table
eval_df = pd.DataFrame(results).T
eval_df = eval_df[['RMSE', 'Precision@5']].round(3)
print("\n--- Model Evaluation Results ---")
print(eval_df)
print("\nProject execution complete.")


