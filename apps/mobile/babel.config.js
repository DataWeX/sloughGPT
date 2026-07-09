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
  ],
};
