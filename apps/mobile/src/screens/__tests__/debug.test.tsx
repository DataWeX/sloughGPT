import {render} from '@testing-library/react-native';
import {View, Text} from 'react-native';
import React from 'react';

test('render returns queries', () => {
  const result = render(<View><Text>hello</Text></View>);
  const keys = Object.keys(result);
  console.log('keys:', keys.slice(0, 10));
  console.log('getByText type:', typeof result.getByText);
  expect(result.getByText('hello')).toBeTruthy();
});
