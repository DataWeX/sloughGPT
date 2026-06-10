# SloughGPT Mobile - Quick Reference

## 🚀 Quick Start Commands

```bash
# Install dependencies
cd apps/mobile && npm install

# Start development
npm start

# Run on iOS
npm run ios

# Run tests
npm test

# Deploy to TestFlight
npm run deploy:testflight
```

## 📱 App Structure

```
apps/mobile/
├── app/              # Screens (Expo Router)
├── components/       # Reusable UI components
├── lib/             # Utilities (API, analytics)
├── stores/          # Zustand state management
├── __tests__/       # Test suite (47 tests)
└── scripts/         # Deployment scripts
```

## 🎨 Theme Colors

| Token | Light | Dark |
|-------|-------|------|
| Primary | #7C52C4 | #C0AAF4 |
| Accent | #EC915F | #F0B082 |
| Success | #34B07D | #48C08C |
| Warning | #ECA83C | #F0C050 |
| Destructive | #DC505A | #EB646E |

## 📊 Analytics Tracking

```typescript
import { Analytics, PerformanceTracker } from '@/lib/analytics'

// Track event
Analytics.trackEvent('feature_used', { feature: 'voice' })

// Track API
await PerformanceTracker.trackApiRequest('/chat', 'POST', fn)

// Track screen
const load = PerformanceTracker.trackScreenLoad('ChatScreen')
load.finish()
```

## 🧪 Testing

```bash
# Run all tests
npm test

# Run once
npm run test:run

# Coverage
npm run test:coverage
```

**Test Results:** 47/47 passing ✅

## 🚀 Deployment

### First Time Setup
1. Install EAS CLI: `npm install -g eas-cli`
2. Login: `eas login`
3. Initialize: `eas init`
4. Configure Sentry DSN in EAS Secrets
5. Update `eas.json` with Apple credentials
6. Run: `npm run deploy:testflight`

### Subsequent Deployments
```bash
npm run deploy:testflight
```

### OTA Updates
```bash
npm run update:production -- --message "Bug fix"
```

## 📚 Documentation

- **README.md** - Project overview
- **SETUP.md** - 10-minute setup guide
- **TESTFLIGHT.md** - Complete deployment guide
- **IMPLEMENTATION_SUMMARY.md** - Full details

## 🔧 Key Files

| File | Purpose |
|------|---------|
| `app/_layout.tsx` | Root layout with Sentry |
| `lib/analytics.ts` | Performance tracking |
| `sentry.config.ts` | Sentry configuration |
| `eas.json` | Build profiles |
| `app.json` | App configuration |

## 🎯 Screens

1. **Login** - Authentication
2. **Chat** - Real-time messaging
3. **Models** - Model management
4. **Knowledge** - Knowledge base
5. **Settings** - Preferences
6. **Health** - System monitoring

## 📦 Stores

1. **auth-store** - Authentication state
2. **chat-store** - Chat and streaming
3. **model-store** - Models and souls
4. **settings-store** - User preferences

## 🔗 Useful Links

- **Expo**: https://docs.expo.dev
- **Tamagui**: https://tamagui.dev
- **Sentry**: https://sentry.io
- **TestFlight**: https://developer.apple.com/testflight/

## 🐛 Troubleshooting

### Build Fails
```bash
eas credentials  # Reset iOS credentials
```

### Tests Fail
```bash
npm run test:run  # Run once to see errors
```

### TypeScript Errors
```bash
npx tsc --noEmit  # Check types
```

### Sentry Not Working
1. Check DSN in EAS Secrets
2. Verify app.json config
3. Check Sentry dashboard

## 📈 Performance Tips

1. Use `PerformanceTracker.trackApiRequest()` for API calls
2. Use `PerformanceTracker.trackScreenLoad()` for screens
3. Track custom events with `Analytics.trackEvent()`
4. Monitor Sentry dashboard regularly

## 🎨 UI Components

```typescript
import { Button, Card, Input, YStack, XStack } from 'tamagui'
import { Icon } from '@tamagui/lucide-icons'

<YStack gap="$3">
  <Card>
    <Text>Content</Text>
  </Card>
  <Button onPress={handlePress}>Click me</Button>
</YStack>
```

## 🔐 Environment Variables

```env
EXPO_PUBLIC_SENTRY_DSN=https://...
EXPO_PUBLIC_API_URL=http://localhost:8000
APP_ENV=development
```

Set in EAS:
```bash
eas secret:create --scope project --name EXPO_PUBLIC_SENTRY_DSN --value "..."
```

## 📱 Device Testing

### iOS Simulator
```bash
npm run ios
```

### Android Emulator
```bash
npm run android
```

### Physical Device
1. Install Expo Go app
2. Run `npm start`
3. Scan QR code

## 🎉 Success Metrics

- ✅ 47 tests passing
- ✅ 0 TypeScript errors
- ✅ TestFlight ready
- ✅ Analytics configured
- ✅ Full documentation

---

**Status: Production Ready** 🚀
