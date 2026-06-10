import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./__tests__/setup.ts'],
    include: ['__tests__/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['__tests__/e2e/**'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        '__tests__/',
        '**/*.d.ts',
        '**/*.config.{ts,js}',
      ],
    },
    deps: {
      inline: [
        'react-native',
        'react-native-web',
        '@react-native-async-storage/async-storage',
        'expo-secure-store',
        'expo-haptics',
        'expo-font',
        'expo-splash-screen',
        'expo-router',
        'tamagui',
        '@tamagui/core',
        '@tamagui/config',
        '@tamagui/lucide-icons',
      ],
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, '.'),
      'react-native': 'react-native-web',
    },
  },
})
