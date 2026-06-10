# TestFlight Deployment Guide

This guide walks you through setting up and deploying SloughGPT Mobile to TestFlight.

## Prerequisites

### 1. Apple Developer Account
- Active Apple Developer Program membership ($99/year)
- Access to [App Store Connect](https://appstoreconnect.apple.com)

### 2. Expo Account
- Free account at [expo.dev](https://expo.dev)
- EAS CLI installed: `npm install -g eas-cli`

### 3. Sentry Account
- Account at [sentry.io](https://sentry.io)
- Create a new React Native project
- Get your DSN from Project Settings → Client Keys

## Initial Setup

### Step 1: Configure EAS

```bash
cd apps/mobile
eas login
eas init
```

This will create an EAS project and add the project ID to `app.json`.

### Step 2: Configure Sentry

1. Update `sentry.config.ts` with your Sentry DSN:
   ```typescript
   dsn: 'https://your-actual-dsn@sentry.io/project-id'
   ```

2. Update `app.json` with your Sentry organization and project:
   ```json
   [
     "@sentry/react-native/expo",
     {
       "organization": "your-org-slug",
       "project": "sloughgpt-mobile"
     }
   ]
   ```

3. Set environment variable for production builds:
   ```bash
   eas secret:create --scope project --name EXPO_PUBLIC_SENTRY_DSN --value "your-sentry-dsn"
   ```

### Step 3: Configure App Store Connect

1. Create a new app in App Store Connect:
   - Bundle ID: `com.sloughgpt.mobile`
   - SKU: `sloughgpt-mobile`
   - Primary Language: English

2. Get your App Store Connect App ID:
   - Go to App Information
   - Copy the "Apple ID" (numeric value)

3. Update `eas.json` with your Apple credentials:
   ```json
   "submit": {
     "production": {
       "ios": {
         "appleId": "your-apple-id@example.com",
         "ascAppId": "123456789",
         "appleTeamId": "ABCD1234"
       }
     }
   }
   ```

### Step 4: Generate iOS Credentials

```bash
eas credentials
```

Follow the prompts to set up:
- Distribution certificate
- Provisioning profile
- Push notifications (optional)

## Building for TestFlight

### Option 1: Using the Deployment Script

```bash
./scripts/deploy-testflight.sh
```

This will:
1. Build the iOS app
2. Submit to App Store Connect
3. Wait for processing

### Option 2: Manual Build

```bash
# Build
eas build --platform ios --profile testflight

# Submit (after build completes)
eas submit --platform ios --profile testflight
```

### Option 3: Build and Submit in One Command

```bash
eas build --platform ios --profile testflight --auto-submit
```

## Managing TestFlight

### Internal Testing

1. Go to App Store Connect → Your App → TestFlight
2. Wait for processing (10-30 minutes)
3. Add internal testers (up to 100 team members)
4. They'll receive an email with install instructions

### External Testing

1. Create a new external testing group
2. Add up to 10,000 testers
3. Submit for Beta App Review (usually 24-48 hours)
4. Once approved, testers can install via TestFlight app

### Monitoring

#### Sentry Dashboard
- **Issues**: View crashes and errors
- **Performance**: Track app performance metrics
- **Releases**: Monitor release health
- **Alerts**: Set up notifications for critical issues

#### App Store Connect
- **Crashes**: View crash reports
- **Analytics**: Track installs, sessions, and retention
- **Feedback**: Read tester feedback

## Version Management

### Incrementing Version

Update `app.json`:
```json
{
  "expo": {
    "version": "1.1.0",
    "ios": {
      "buildNumber": "2"
    }
  }
}
```

Or use auto-increment (already configured in `eas.json`):
```bash
eas build --platform ios --profile testflight --auto-increment
```

### OTA Updates (Expo Updates)

For JavaScript-only changes (no native code changes):

```bash
# Publish update
eas update --branch production --message "Fix chat bug"

# Check update status
eas update:list
```

Users will receive updates automatically on app launch.

## Troubleshooting

### Build Failures

**Issue**: "No provisioning profile found"
```bash
eas credentials
# Select iOS → Remove profile → Rebuild
```

**Issue**: "Certificate expired"
```bash
eas credentials
# Select iOS → Remove certificate → Create new
```

### Submission Failures

**Issue**: "Invalid API key"
```bash
# Re-authenticate with App Store Connect
eas submit --platform ios
# Select "Log in to your Apple Developer account"
```

**Issue**: "Missing export compliance"
- Already configured in `app.json` with `ITSAppUsesNonExemptEncryption: false`
- If you use encryption, update this value and provide documentation

### Sentry Issues

**Issue**: "Source maps not uploaded"
```bash
# Manually upload source maps
npx sentry-react-native upload-debug-symbols \
  --org your-org \
  --project sloughgpt-mobile
```

## Performance Monitoring

### Key Metrics to Track

1. **Crash-free rate**: Should be >99%
2. **App launch time**: Should be <2 seconds
3. **API response time**: Track in Sentry Performance
4. **Memory usage**: Monitor for leaks
5. **Battery usage**: Check in Xcode Instruments

### Custom Metrics

Add custom metrics in your code:

```typescript
import { PerformanceTracker } from '../lib/analytics'

// Track API performance
await PerformanceTracker.trackApiRequest('/chat/stream', 'POST', async () => {
  return await chatController.sendMessage(message)
})

// Track screen load
const screenLoad = PerformanceTracker.trackScreenLoad('ChatScreen')
// ... load data
screenLoad.finish()

// Track custom metrics
PerformanceTracker.trackMetric('messages_sent', 1, { model: 'gpt2' })
```

## Best Practices

### Before Each Release

1. ✅ Run all tests: `npm run test:run`
2. ✅ Test on physical device
3. ✅ Check Sentry for unresolved issues
4. ✅ Update version number
5. ✅ Write release notes

### Release Notes Template

```
Version 1.1.0

✨ New Features
- Added voice input for chat
- Dark mode support

🐛 Bug Fixes
- Fixed chat streaming on slow connections
- Resolved memory leak in model switching

⚡️ Performance
- 30% faster app launch
- Reduced memory usage by 15%
```

## Support

- **Expo Documentation**: https://docs.expo.dev
- **EAS Build**: https://docs.expo.dev/build/introduction/
- **TestFlight**: https://developer.apple.com/testflight/
- **Sentry**: https://docs.sentry.io/platforms/react-native/

## Quick Commands

```bash
# Build for TestFlight
eas build --platform ios --profile testflight

# Submit to TestFlight
eas submit --platform ios --profile testflight

# Build and submit
eas build --platform ios --profile testflight --auto-submit

# Publish OTA update
eas update --branch production --message "Bug fix"

# Check build status
eas build:list

# View credentials
eas credentials

# Increment version
npm version patch  # or minor, major
```
