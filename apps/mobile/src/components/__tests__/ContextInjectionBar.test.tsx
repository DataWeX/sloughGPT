import React from 'react';
import {render, fireEvent} from '@testing-library/react-native';
import {TamaguiProvider} from 'tamagui';
import appConfig from '../../../tamagui.config';
import {ContextInjectionBar} from '../ContextInjectionBar';

const Wrapper = ({children}: any) => (
  <TamaguiProvider config={appConfig} defaultTheme="light">{children}</TamaguiProvider>
);

async function renderBar(props: Partial<React.ComponentProps<typeof ContextInjectionBar>> = {}) {
  return render(
    <ContextInjectionBar onInject={jest.fn()} {...props} />,
    {wrapper: Wrapper},
  );
}

describe('ContextInjectionBar', () => {
  it('renders the + Context trigger', async () => {
    const {getByTestId} = await renderBar();
    expect(getByTestId('inject-context-trigger')).toBeTruthy();
  });

  it('opens the input when trigger pressed', async () => {
    const {getByTestId, queryByTestId} = await renderBar();
    await fireEvent.press(getByTestId('inject-context-trigger'));
    expect(queryByTestId('inject-context-trigger')).toBeNull();
    expect(getByTestId('context-input')).toBeTruthy();
  });

  it('injects trimmed text on Send and closes', async () => {
    const onInject = jest.fn();
    const {getByTestId, getByText, queryByTestId} = await renderBar({onInject});
    await fireEvent.press(getByTestId('inject-context-trigger'));
    await fireEvent.changeText(getByTestId('context-input'), '  extra rules  ');
    await fireEvent.press(getByText('Send'));
    expect(onInject).toHaveBeenCalledWith('extra rules');
    expect(queryByTestId('context-input')).toBeNull();
  });

  it('does not inject empty text', async () => {
    const onInject = jest.fn();
    const {getByTestId, getByText} = await renderBar({onInject});
    await fireEvent.press(getByTestId('inject-context-trigger'));
    await fireEvent.changeText(getByTestId('context-input'), '   ');
    await fireEvent.press(getByText('Send'));
    expect(onInject).not.toHaveBeenCalled();
    expect(getByTestId('context-input')).toBeTruthy();
  });

  it('cancels without injecting', async () => {
    const onInject = jest.fn();
    const {getByTestId, getByText, queryByTestId} = await renderBar({onInject});
    await fireEvent.press(getByTestId('inject-context-trigger'));
    await fireEvent.changeText(getByTestId('context-input'), 'draft');
    await fireEvent.press(getByText('Cancel'));
    expect(onInject).not.toHaveBeenCalled();
    expect(queryByTestId('context-input')).toBeNull();
  });

  it('respects disabled', async () => {
    const {getByTestId} = await renderBar({disabled: true});
    expect(getByTestId('inject-context-trigger').props.accessibilityState?.disabled).toBe(true);
  });
});
