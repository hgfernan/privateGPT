#! /bin/sh 

echo "$0": Starting the initial setup 

# Copy default environment file to the default environment
cp example.env .env

# Create folders for the model and the database
mkdir models db

# Download the default model to its database
cd models/
wget -c https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin

# Will ingest information in the database

echo "$0": Will ingest the source information in the database 
./ingest.py 2>&1 | tee ingest.out

echo "$0": Finished the initial setup 
