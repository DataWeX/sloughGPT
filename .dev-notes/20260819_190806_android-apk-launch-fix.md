---
id: 20260819_190806_android-apk-launch-fix
title: Android APK launch fix
status: done
tags: android,mobile,build
created: 2026-08-19T19:08:06.875400+00:00
---

Android APK launch fix

Fixed 4 cascading Android APK launch crashes: (1) abiFilters x86_64-only had no ARM code - fixed to arm64-v8a+armeabi-v7a+x86_64; (2) isNewArchEnabled override removed (unsupported since RN 0.82); (3) ReactInstanceManager.createReactContext removed (bridge mode dropped in RN 0.86); (4) librninstance.so missing - stub created in jniLibs. Removed explicit DefaultNewArchitectureEntryPoint.load() from MainApplication. Enabled fabricEnabled=true. Bundled JS via react-native bundle. App runs: onboarding → chat screen with suggestion chips, input bar, bottom tabs. 90/90 test suites, 843/843 tests pass.