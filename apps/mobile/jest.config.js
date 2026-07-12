module.exports = {
  preset: '@react-native/jest-preset',
  setupFiles: ['<rootDir>/jest-setup.js'],
  testTimeout: 15000,
  transformIgnorePatterns: [
    'node_modules/(?!(@react-native|react-native|zustand|@react-navigation|react-native-safe-area-context|react-native-screens|@react-native-async-storage|react-native-markdown-display|tamagui|@tamagui)/)',
  ],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^react-native-fs$': '<rootDir>/src/__mocks__/react-native-fs.ts',
  },
};
