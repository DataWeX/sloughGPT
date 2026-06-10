#!/bin/bash

# TestFlight Deployment Script for SloughGPT Mobile
# This script builds the iOS app and submits it to TestFlight

set -e  # Exit on error

echo "🚀 Starting TestFlight deployment..."

# Check if EAS CLI is installed
if ! command -v eas &> /dev/null; then
    echo "❌ EAS CLI not found. Installing..."
    npm install -g eas-cli
fi

# Check if logged in to EAS
if ! eas whoami &> /dev/null; then
    echo "❌ Not logged in to EAS. Please run: eas login"
    exit 1
fi

# Get current version
VERSION=$(node -p "require('./app.json').expo.version")
BUILD_NUMBER=$(node -p "require('./app.json').expo.ios.buildNumber")

echo "📱 Building version $VERSION (build $BUILD_NUMBER)..."

# Build for TestFlight
echo "🔨 Building iOS app for TestFlight..."
eas build --platform ios --profile testflight --non-interactive --wait

echo "✅ Build complete!"

# Submit to TestFlight
echo "📤 Submitting to TestFlight..."
eas submit --platform ios --profile testflight --non-interactive --wait

echo "✅ Successfully submitted to TestFlight!"
echo ""
echo "📊 Next steps:"
echo "1. Go to App Store Connect"
echo "2. Wait for processing to complete (usually 10-30 minutes)"
echo "3. Add internal testers or submit for external review"
echo "4. Monitor crash reports in Sentry dashboard"
echo ""
echo "🔗 Useful links:"
echo "- App Store Connect: https://appstoreconnect.apple.com"
echo "- Sentry Dashboard: https://sentry.io"
echo "- EAS Build: https://expo.dev"
