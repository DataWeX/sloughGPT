module.exports = {
  preset: '@react-native/jest-preset',
  setupFiles: ['<rootDir>/jest-setup.js'],
  testTimeout: 15000,
  transformIgnorePatterns: [
    'node_modules/(?!(@react-native|react-native|zustand|@react-navigation|react-native-safe-area-context|react-native-screens|@react-native-async-storage|react-native-markdown-display|tamagui|@tamagui|lucide-react-native|react-native-svg)/)',
  ],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^react-native-fs$': '<rootDir>/src/__mocks__/react-native-fs.ts',
    '^expo-image-picker$': '<rootDir>/src/__mocks__/expo-image-picker.ts',
    '^expo-document-picker$': '<rootDir>/src/__mocks__/expo-document-picker.ts',
    '^expo-file-system$': '<rootDir>/src/__mocks__/expo-file-system.ts',
  },
};
