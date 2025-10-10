#!/bin/bash

echo "🚀 Starting Pharmacology Graph App"
echo "=================================="
echo ""

# Kill any existing instances
pkill -f "python.*app.py" 2>/dev/null
sleep 1

# Clear port if needed
lsof -ti:7860 | xargs kill -9 2>/dev/null
sleep 1

echo "✓ Port 7860 ready"
echo "✓ Starting app..."
echo ""
echo "Open in browser: http://localhost:7860"
echo "Press Ctrl+C to stop"
echo ""

# Start the app
cd /home/joe/projects/pharmacology-graph
python app.py
