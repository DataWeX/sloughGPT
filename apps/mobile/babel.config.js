module.exports = {
  presets: ['module:@react-native/babel-preset'],
  plugins: [
    [
      '@tamagui/babel-plugin',
      {
        components: ['tamagui'],
        config: './tamagui.config.ts',
      },
    ],
    // Strip console.log/debug in release builds
    ...(process.env.NODE_ENV === 'production'
      ? [['transform-remove-console', {exclude: ['error', 'warn']}]]
      : []),
  ],
};
