module.exports = {
  preset: '@react-native/jest-preset',
  setupFiles: [],
  transformIgnorePatterns: [
    'node_modules/(?!(@react-native|react-native|zustand|@react-navigation|react-native-safe-area-context|react-native-screens|@react-native-async-storage|react-native-markdown-display)/)',
  ],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
};
