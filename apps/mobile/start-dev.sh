#!/bin/bash

# SloughGPT Mobile - Development Startup Script
# This script starts both the backend API and mobile app

echo "🚀 Starting SloughGPT Mobile Development Environment"
echo "=================================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
lsof -ti :8000 | xargs kill -9 2>/dev/null
lsof -ti :8081 | xargs kill -9 2>/dev/null
sleep 2

# Start backend
echo ""
echo "📡 Starting backend API server..."
cd /Users/mac/sloughGPT/apps/api/server
MAN_AUTOLOAD_MODEL="" python3 main.py > /tmp/sloughgpt-backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend
sleep 5
if lsof -ti :8000 > /dev/null 2>&1; then
    echo -e "   ${GREEN}✅ Backend running on http://localhost:8000${NC}"
else
    echo -e "   ${RED}❌ Backend failed to start${NC}"
    echo "   Check log: tail -50 /tmp/sloughgpt-backend.log"
fi

# Start mobile app
echo ""
echo "📱 Starting mobile app..."
cd /Users/mac/sloughGPT/apps/mobile
npx expo start --clear --port 8081 > /tmp/expo-dev.log 2>&1 &
EXPO_PID=$!
echo "   Expo PID: $EXPO_PID"

# Wait for Expo
sleep 8
if lsof -ti :8081 > /dev/null 2>&1; then
    echo -e "   ${GREEN}✅ Expo running on http://localhost:8081${NC}"
else
    echo -e "   ${RED}❌ Expo failed to start${NC}"
    echo "   Check log: tail -50 /tmp/expo-dev.log"
fi

echo ""
echo "=================================================="
echo -e "${YELLOW}📲 To test on your iPhone:${NC}"
echo "   1. Install 'Expo Go' from the App Store"
echo "   2. Make sure your phone is on the same WiFi as your computer"
echo "   3. Open the Expo Go app"
echo "   4. Scan the QR code from the terminal running 'npm start'"
echo ""
echo -e "${YELLOW}📊 Monitoring:${NC}"
echo "   Backend logs: tail -f /tmp/sloughgpt-backend.log"
echo "   Expo logs:     tail -f /tmp/expo-dev.log"
echo ""
echo -e "${YELLOW}🛑 To stop:${NC}"
echo "   pkill -f 'python3 main.py'"
echo "   pkill -f 'expo start'"
echo ""
echo "=================================================="
