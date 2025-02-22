# analysis

## To run:

Setup a local postgres database and [install pgvector](https://github.com/pgvector/pgvector?tab=readme-ov-file#installation).

Create the tables:

```
create table job_clusters (cluster_id integer,job_title varchar, provider varchar, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP)
```

```
create table job_title_embeddings (job_title varchar, embedding vector(1536), provider varchar, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP)
```

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```
OPENAI_API_KEY=<your-key> ONET_FILE=job-title-data/onet-occupation-alt-data.txt DB_NAME=<your-db-name> DB_USER=<your-db-user> DB_PASSWORD=<your-db-password> python cluster_onet_job_titles.py
```
