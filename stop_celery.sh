#!/bin/bash

# Stop Celery worker and beat scheduler

echo "Stopping Celery services..."

# Stop Celery beat
echo "Stopping Celery beat..."
pkill -f "celery.*beat"

# Stop Celery worker
echo "Stopping Celery worker..."
pkill -f "celery.*worker"

echo "Celery services stopped."