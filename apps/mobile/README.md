# SloughGPT Mobile

A beautiful, production-ready React Native mobile app for SloughGPT with real-time chat, model management, and comprehensive analytics.

![React Native](https://img.shields.io/badge/React%20Native-0.85.3-blue)
![Expo](https://img.shields.io/badge/Expo-56-black)
![TypeScript](https://img.shields.io/badge/TypeScript-6.0-blue)
![Tests](https://img.shields.io/badge/Tests-47%20passing-green)

## ✨ Features

### Core Functionality
- 💬 **Real-time Chat** - SSE streaming with markdown support
- 🎤 **Voice Input** - Record and transcribe voice messages
- 📷 **Image Upload** - Capture photos or select from library
- 🤖 **Model Management** - Load/switch models, manage souls and checkpoints
- 📚 **Knowledge Base** - Search, filter, and manage knowledge items
- ⚙️ **Settings** - Theme customization, chat defaults, system health
- 🔐 **Authentication** - Secure login with Expo SecureStore
- 🌙 **Dark Mode** - Full light/dark theme support

### Production Features
- 📊 **Analytics** - Sentry crash reporting and performance monitoring
- 🚀 **TestFlight Ready** - One-command deployment to iOS TestFlight
- 🔄 **OTA Updates** - Push JavaScript updates without App Store review
- 📱 **Native Performance** - Built with React Native and Tamagui
- 🧪 **Comprehensive Tests** - 47 unit and integration tests

## 🚀 Quick Start

### Prerequisites
- Node.js 20+
- npm or yarn
- iOS Simulator (macOS) or Android Emulator
- Expo Go app (for physical device testing)

### Installation

```bash
cd apps/mobile
npm install
```

### Development

```bash
# Start development server
npm start

# Run on iOS
npm run ios

# Run on Android
npm run android

# Run tests
npm test
```

### Deployment to TestFlight

**First time setup?** See [SETUP.md](./SETUP.md) for detailed instructions.

```bash
# One-command deployment
npm run deploy:testflight

# Or step by step
npm run build:ios
npm run submit:ios
```

## 📁 Project Structure

```
apps/mobile/
├── app/                    # Expo Router screens
│   ├── (auth)/            # Authentication screens
│   │   └── login.tsx
│   ├── (tabs)/            # Main tab navigation
│   │   ├── chat/          # Chat screen with streaming
│   │   ├── models/        # Model management
│   │   ├── knowledge/     # Knowledge base
│   │   └── settings/      # Settings + Health
│   └── _layout.tsx        # Root layout with Sentry
├── components/            # Reusable components
├── lib/                   # Utilities
│   ├── api-client.ts      # API client with retry logic
│   ├── sse-client.ts      # SSE streaming parser
│   ├── analytics.ts       # Performance & analytics tracking
│   └── config.ts          # App configuration
├── stores/                # Zustand state management
│   ├── auth-store.ts      # Authentication state
│   ├── chat-store.ts      # Chat state and streaming
│   ├── model-store.ts     # Models, souls, checkpoints
│   └── settings-store.ts  # User preferences
├── hooks/                 # Custom React hooks
├── assets/                # Fonts and images
├── __tests__/             # Test suite (47 tests)
├── scripts/               # Deployment scripts
├── tamagui.config.ts      # Theme configuration
├── sentry.config.ts       # Sentry configuration
├── eas.json               # EAS Build configuration
└── package.json
```

## 🎨 Design System

### Noir Violet Theme

| Token | Light | Dark | Usage |
|-------|-------|------|-------|
| Primary | `#7C52C4` | `#C0AAF4` | Main actions, links |
| Accent | `#EC915F` | `#F0B082` | Highlights, CTAs |
| Success | `#34B07D` | `#48C08C` | Positive feedback |
| Warning | `#ECA83C` | `#F0C050` | Caution states |
| Destructive | `#DC505A` | `#EB646E` | Errors, delete |

### Typography
- **Sans**: Outfit (400, 500, 600, 700)
- **Mono**: JetBrains Mono (400, 500)

## 📊 Analytics & Performance

### Automatic Tracking
- App crashes and errors
- Performance metrics (launch time, frame rates)
- Network requests and response times
- Memory usage and leaks
- User sessions and navigation

### Custom Tracking

```typescript
import { Analytics, PerformanceTracker } from '@/lib/analytics'

// Track custom event
Analytics.trackEvent('feature_used', { feature: 'voice_input' })

// Track API performance
await PerformanceTracker.trackApiRequest('/chat/stream', 'POST', async () => {
  return await chatController.sendMessage(message)
})

// Track screen load
const screenLoad = PerformanceTracker.trackScreenLoad('ChatScreen')
// ... load data
screenLoad.finish()
```

### Sentry Dashboard
- **Issues**: View crashes and errors in real-time
- **Performance**: Track app performance metrics
- **Releases**: Monitor release health and adoption
- **Alerts**: Get notified of critical issues

## 🧪 Testing

```bash
# Run all tests
npm test

# Run tests once (CI mode)
npm run test:run

# Run tests with coverage
npm run test:coverage

# Run tests with UI
npm run test:ui
```

### Test Coverage
- ✅ 47 tests passing
- ✅ Unit tests for all stores
- ✅ Integration tests for chat flow
- ✅ API client and SSE parser tests
- ✅ Component smoke tests

## 🚀 Deployment

### TestFlight (iOS)

```bash
# Build and submit
npm run deploy:testflight

# Or manually
npm run build:ios
npm run submit:ios
```

See [TESTFLIGHT.md](./TESTFLIGHT.md) for detailed deployment guide.

### OTA Updates

```bash
# Publish JavaScript update
npm run update:production -- --message "Fix chat bug"

# Check update status
eas update:list
```

### Version Management

```bash
# Increment version
npm version patch  # 1.0.0 → 1.0.1
npm version minor  # 1.0.0 → 1.1.0
npm version major  # 1.0.0 → 2.0.0

# Build with auto-increment
npm run deploy:testflight
```

## 🔧 Configuration

### Environment Variables

Create `.env` in `apps/mobile/`:

```env
EXPO_PUBLIC_SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
EXPO_PUBLIC_API_URL=http://localhost:8000
APP_ENV=development
```

See `.env.example` for all available options.

### App Configuration

Edit `app.json` for:
- App name and slug
- Bundle identifiers
- Splash screen
- App icons
- Permissions

## 📱 Screenshots

| Chat | Models | Knowledge | Settings |
|------|--------|-----------|----------|
| [Chat Screen] | [Models Screen] | [Knowledge Screen] | [Settings Screen] |

## 🏗️ Tech Stack

- **Framework**: React Native 0.85 + Expo SDK 56
- **UI Library**: Tamagui (Noir Violet theme)
- **Navigation**: Expo Router (file-based routing)
- **State Management**: Zustand
- **API Client**: Custom fetch-based with retry logic
- **Streaming**: SSE client for real-time chat
- **Testing**: Vitest + React Native Testing Library
- **Analytics**: Sentry (crash reporting + performance)
- **Deployment**: EAS Build + TestFlight

## 🤝 Contributing

1. Follow the existing code style (Prettier configured)
2. Write tests for new features
3. Update documentation
4. Test on both iOS and Android
5. Add analytics tracking for new features

## 📚 Documentation

- [SETUP.md](./SETUP.md) - Quick setup guide (10 minutes)
- [TESTFLIGHT.md](./TESTFLIGHT.md) - Complete TestFlight deployment guide
- [__tests__/TEST_SUMMARY.md](__tests__/TEST_SUMMARY.md) - Test suite documentation

## 🔗 Links

- **Backend API**: [apps/api/server/](../api/server/)
- **Web App**: [apps/web/](../web/)
- **Expo Documentation**: https://docs.expo.dev
- **Tamagui**: https://tamagui.dev
- **Sentry**: https://sentry.io

## 📄 License

Part of the SloughGPT monorepo.

## 🆘 Support

- **Issues**: Open an issue on GitHub
- **Expo Discord**: https://chat.expo.dev
- **Sentry Support**: https://sentry.io/support/

---

Built with ❤️ using React Native, Expo, and Tamagui
