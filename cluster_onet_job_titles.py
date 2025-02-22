import csv
import openai
import psycopg2
from datetime import datetime
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os

from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn import metrics
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

openai_api_key = os.environ.get('OPENAI_API_KEY')

# Database connection parameters
db_params = {
    'dbname': os.environ.get('DB_NAME'),
    'user': os.environ.get('DB_USER'),
    'password': os.environ.get('DB_PASSWORD', ''),
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432')
}

client = openai.OpenAI(
    api_key=openai_api_key
)

onet_file = os.environ.get('ONET_FILE')

EPS_MIN_SAMPLES = 2

def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(
            input=[f"{text} (This is a job title held by someone looking for another job with similar skills, roles, and responsibilities.)"],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for {text}: {e}")
        return None

def delete_all_embeddings():
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    cursor.execute('delete from job_title_embeddings')
    conn.commit()
    cursor.close()
    conn.close()

# Function to write embedding to PostgreSQL
def write_to_db(job_title, embedding):
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    timestamp = datetime.now()
    try:
        cursor.execute("""
            INSERT INTO job_title_embeddings (job_title, embedding, provider, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (job_title, embedding, 'openai', timestamp, timestamp))
    except:
        print(f"Error writing {job_title} to the database")
    conn.commit()
    cursor.close()
    conn.close()

def get_provider_embeddings(provider):
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    cursor.execute("select job_title, embedding from job_title_embeddings where provider = %s", (provider,))

    return cursor.fetchall()

def pca_variance(embeddings, plot=True):
    pca = PCA().fit(embeddings)

    if plot:
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()

def elbow(embeddings):
    k = EPS_MIN_SAMPLES
    nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Sort and plot k-th nearest neighbor distances
    distances = np.sort(distances[:, k - 1])
    plt.plot(distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{k}-th Nearest Neighbor Distance")
    plt.title("Elbow Method for Selecting eps")
    plt.show()

def pca(embeddings):
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    pca = PCA(n_components=270)
    embeddings_pca = pca.fit_transform(embeddings_scaled)

    return embeddings_pca

def dbscan(embeddings):
    clustering = DBSCAN(eps=21, min_samples=EPS_MIN_SAMPLES).fit(embeddings)

    return clustering.labels_.tolist()

def hdbscan(embeddings):
    # TODO: cosine metric
    clustering = HDBSCAN(min_cluster_size=2, metric='euclidean').fit(embeddings)

    return clustering.labels_.tolist()

def dendrogram(embeddings):
    Z = linkage(embeddings, method='ward')
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.axhline(y=19, color='r', linestyle='--')  # Visualize chosen t
    plt.show()

def hac(embeddings):
    Z = linkage(embeddings, method='ward')
    clusters = fcluster(Z, t=19, criterion='distance')
    return clusters.tolist()

def save_results(embeddings, clusters, provider, delete_existing=True):
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    if delete_existing:
        cursor.execute('delete from job_clusters where provider = %s', (provider,))

    for i, c in enumerate(clusters):
        cursor.execute('insert into job_clusters (cluster_id,job_title,provider,created_at,updated_at) values (%s,%s,%s,%s,%s)', (c, embeddings[i][0], provider, datetime.now(), datetime.now()))

    conn.commit()
    cursor.close()
    conn.close()

def parse_onet_and_retrieve_embeddings():
    job_titles = []

    # Read the tab-delimited text file
    with open(onet_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip the first header row
        for row in reader:
            job_title = row[1]  # Assuming the second column is the job title
            job_titles.append(job_title)
    
    # deduplicate job titles
    job_titles = list(set(job_titles))

    for job_title in job_titles:
        embedding = get_embedding(job_title)
        write_to_db(job_title, embedding)


if __name__ == "__main__":
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    #!!! make sure the existing table is dumped before uncommenting this!!!
    # delete_all_embeddings()

    parse_onet_and_retrieve_embeddings()

    job_embeddings = get_provider_embeddings('openai')

    embeddings = [json.loads(embedding[1]) for embedding in job_embeddings]

    pca_variance(embeddings, False)
    embeddings = np.array(embeddings)  # Convert embeddings list to numpy array
    embeddings_pca = pca(embeddings)
    elbow(embeddings_pca) # for dbscan
    clusters = dbscan(embeddings_pca)

    # print("Drawing dendogram")
    # dendrogram(embeddings_pca)

    save_results(job_embeddings, clusters, 'openai')

    print("Embeddings have been written to the database.")
