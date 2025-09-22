#!/bin/bash

# Celery Worker and Beat startup script for Improv Today simulation engine

echo "Starting Celery worker and beat scheduler for simulation engine..."

# Set environment variables
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Start Celery worker in background
echo "Starting Celery worker..."
celery -A celery_app worker --loglevel=info --queues=simulation --detach

# Start Celery beat scheduler in background
echo "Starting Celery beat scheduler..."
celery -A celery_app beat --loglevel=info --detach

echo "Celery services started successfully!"
echo "Monitor with: celery -A celery_app events"
echo "Stop with: ./stop_celery.sh"