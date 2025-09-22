#!/bin/bash

# EC2 Deployment Script for Improv Today Backend

echo "🚀 Deploying Improv Today Backend to EC2..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Create app directory
sudo mkdir -p /opt/improv-today
cd /opt/improv-today

# Copy files (assuming you've uploaded them)
echo "📁 Files should be uploaded to /opt/improv-today/"

# Set up environment
cp .env.production.example .env.production
echo "⚠️  Update .env.production with your actual values!"

# Build and start services
echo "🐳 Starting Docker services..."
docker-compose -f docker-compose.prod.yml up -d --build

# Show status
echo "✅ Deployment complete!"
echo "🔍 Check status: docker-compose -f docker-compose.prod.yml ps"
echo "📋 Check logs: docker-compose -f docker-compose.prod.yml logs -f"