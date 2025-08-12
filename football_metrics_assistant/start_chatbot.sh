#!/bin/bash

# Football Metrics Assistant Chatbot Startup Script
# This script starts the chatbot server with proper environment setup

echo "⚽ Football Metrics Assistant Chatbot"
echo "======================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3 and try again"
    exit 1
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python3 -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Required packages not found. Installing..."
    pip3 install fastapi uvicorn
fi

# Check if we're in the right directory
if [ ! -f "chatbot_server.py" ]; then
    echo "❌ Please run this script from the football_metrics_assistant directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check for Google API key
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️  Warning: GOOGLE_API_KEY environment variable not set"
    echo "   The chatbot will work but won't have AI-powered responses"
    echo "   Set it with: export GOOGLE_API_KEY='your_api_key_here'"
    echo ""
fi

# Start the chatbot server
echo "🚀 Starting Football Metrics Assistant Chatbot..."
echo "📱 Chatbot will be available at: http://localhost:8080"
echo "🔌 API will be available at: http://localhost:8080/api"
echo "📊 Health check at: http://localhost:8080/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 chatbot_server.py 