#!/bin/bash
source venv/Scripts/activate

echo "Starting Docker Desktop"
# Memanggil PowerShell Windows dari dalam WSL untuk start Docker
powershell.exe -Command "Start-Process 'C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe'"

echo "Waiting for Docker Engine"
# Loop cek sampai docker info tidak error
while ! docker info > /dev/null 2>&1; do
  echo -n "."
  sleep 2
done
echo ""
echo "Docker is ready"

echo "Turning on Remosy Container"
docker compose up -d

echo "---------------------------------------------------"
echo "System is running"
echo "---------------------------------------------------"
echo "Airflow Webserver : http://localhost:8080"
echo "Streamlit Dashboard: http://localhost:8501"
echo "Database Warehouse : http://localhost:5434 (Via DBeaver)"
echo "---------------------------------------------------"
