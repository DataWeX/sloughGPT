# Quick Setup Guide - TestFlight & Analytics

This guide will get you up and running with TestFlight deployment and performance tracking in 10 minutes.

## 🚀 Quick Start

### 1. Install EAS CLI
```bash
npm install -g eas-cli
```

### 2. Login to Expo
```bash
cd apps/mobile
eas login
```

### 3. Initialize EAS Project
```bash
eas init
```
This will prompt you to create a new project and add the project ID to `app.json`.

### 4. Set Up Sentry (Crash Reporting & Analytics)

**a. Create Sentry Account**
- Go to https://sentry.io and sign up (free tier available)
- Create a new project: React Native
- Copy your DSN from Project Settings → Client Keys

**b. Add Sentry DSN to EAS Secrets**
```bash
eas secret:create --scope project --name EXPO_PUBLIC_SENTRY_DSN --value "your-sentry-dsn-here"
```

**c. Update app.json**
Replace the Sentry plugin configuration with your org and project:
```json
[
  "@sentry/react-native/expo",
  {
    "organization": "your-org-slug",
    "project": "sloughgpt-mobile"
  }
]
```

### 5. Set Up Apple Developer Account

**a. Prerequisites**
- Apple Developer Program membership ($99/year)
- Access to App Store Connect

**b. Create App in App Store Connect**
1. Go to https://appstoreconnect.apple.com
2. Apps → + → New App
3. Fill in:
   - Platform: iOS
   - Name: SloughGPT
   - Primary Language: English
   - Bundle ID: `com.sloughgpt.mobile` (create if needed)
   - SKU: `sloughgpt-mobile`
4. Copy the **Apple ID** (numeric) from App Information

**c. Get Your Team ID**
1. Go to https://developer.apple.com/account
2. Membership details → Team ID (e.g., `ABCD1234`)

**d. Update eas.json**
```json
"submit": {
  "production": {
    "ios": {
      "appleId": "your-email@example.com",
      "ascAppId": "1234567890",
      "appleTeamId": "ABCD1234"
    }
  }
}
```

### 6. Configure iOS Credentials
```bash
eas credentials
```
Select:
- iOS → Production: Manage your certificates and provisioning profiles
- Let EAS handle everything automatically (recommended)

### 7. Build and Submit to TestFlight

**Option A: One Command**
```bash
npm run deploy:testflight
```

**Option B: Step by Step**
```bash
# Build
npm run build:ios

# Submit (after build completes)
npm run submit:ios
```

**Option C: Build + Auto Submit**
```bash
eas build --platform ios --profile testflight --auto-submit
```

### 8. Monitor Performance

**Sentry Dashboard**
- Issues: View crashes and errors
- Performance: Track app performance
- Releases: Monitor release health
- Alerts: Set up notifications

**App Store Connect**
- TestFlight: Manage testers
- Analytics: Track installs and usage
- Crashes: View crash reports

## 📊 What Gets Tracked

### Automatic Tracking
- ✅ App crashes and errors
- ✅ Performance metrics (launch time, frame rates)
- ✅ Network requests and response times
- ✅ Memory usage and leaks
- ✅ User sessions and navigation

### Custom Tracking (Already Implemented)
- ✅ Chat messages sent (with model/soul metadata)
- ✅ Session creation
- ✅ Message feedback (thumbs up/down)
- ✅ Message regeneration
- ✅ Screen load times
- ✅ API request performance

### Adding Custom Tracking
```typescript
import { Analytics, PerformanceTracker } from '@/lib/analytics'

// Track custom event
Analytics.trackEvent('feature_used', { feature: 'voice_input' })

// Track API performance
await PerformanceTracker.trackApiRequest('/models', 'GET', async () => {
  return await fetchModels()
})

// Track screen load
const screenLoad = PerformanceTracker.trackScreenLoad('ModelsScreen')
// ... load data
screenLoad.finish()
```

## 🧪 Testing

### Internal Testing (Immediate)
1. Build completes → appears in App Store Connect → TestFlight
2. Wait for processing (10-30 minutes)
3. Add internal testers (up to 100 team members)
4. They receive email with install link

### External Testing (Requires Review)
1. Create external testing group
2. Add up to 10,000 testers
3. Submit for Beta App Review (24-48 hours)
4. Once approved, testers install via TestFlight app

## 🔄 OTA Updates (JavaScript Only)

For quick fixes without App Store review:

```bash
# Publish update
npm run update:production -- --message "Fix chat bug"

# Check update status
eas update:list
```

Users receive updates automatically on app launch.

## 📱 Version Management

### Increment Version
```bash
# Patch (1.0.0 → 1.0.1)
npm version patch

# Minor (1.0.0 → 1.1.0)
npm version minor

# Major (1.0.0 → 2.0.0)
npm version major
```

Then rebuild:
```bash
npm run deploy:testflight
```

### Auto-Increment Build Number
Already configured in `eas.json` - build numbers increment automatically.

## 🐛 Troubleshooting

### Build Fails: "No provisioning profile"
```bash
eas credentials
# Select iOS → Remove profile → Rebuild
```

### Submission Fails: "Invalid API key"
```bash
eas submit --platform ios
# Select "Log in to your Apple Developer account"
```

### Sentry Not Receiving Events
1. Check DSN is set: `eas secret:list`
2. Verify app.json has correct org/project
3. Check Sentry dashboard for errors

### App Crashes on Launch
1. Check Sentry dashboard for crash reports
2. Test locally: `npm run ios`
3. Check device logs in Xcode

## 📚 Resources

- **Expo EAS**: https://docs.expo.dev/build/introduction/
- **TestFlight**: https://developer.apple.com/testflight/
- **Sentry React Native**: https://docs.sentry.io/platforms/react-native/
- **App Store Connect**: https://appstoreconnect.apple.com

## ✅ Checklist

Before your first release:

- [ ] EAS CLI installed and logged in
- [ ] Sentry account created and DSN configured
- [ ] Apple Developer account active
- [ ] App created in App Store Connect
- [ ] iOS credentials configured in EAS
- [ ] Build successful
- [ ] Submitted to TestFlight
- [ ] Internal testers added
- [ ] Sentry dashboard monitoring
- [ ] Release notes written

## 🎯 Next Steps

1. **Add More Testers**: Expand internal testing group
2. **Set Up Alerts**: Configure Sentry alerts for critical issues
3. **Monitor Performance**: Check Sentry Performance tab regularly
4. **Gather Feedback**: Use TestFlight's feedback feature
5. **Plan Release**: Prepare for App Store submission

## 🆘 Support

- **Expo Discord**: https://chat.expo.dev
- **Sentry Support**: https://sentry.io/support/
- **Apple Developer**: https://developer.apple.com/support/

---

**Need help?** Check the full guide: [TESTFLIGHT.md](./TESTFLIGHT.md)
