#!/bin/sh

PROJECT_DIR="/mapreduce-test/mapreduce-project1/big_data_project_1"
DATA_DIR="$PROJECT_DIR/data"

# === Prep directory ===
echo "Creating data directory if not already present..."
mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || { echo "Failed data directory access"; exit 1; }

# === Download CSV using curl and API token ===
APP_TOKEN="tzksBXvIq9l9Ub985hzK3D5HP"
API_URL="https://data.cityofnewyork.us/resource/pvqr-7yc4.csv?\$limit=10000000"

curl -v -H "X-App-Token: $APP_TOKEN" "$API_URL" -o "$DATA_DIR/data.csv"