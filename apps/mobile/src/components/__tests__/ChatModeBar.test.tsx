import React from 'react';
import {render, screen, fireEvent, act, cleanup} from '@testing-library/react-native';
import {TamaguiProvider} from 'tamagui';
import appConfig from '../../../tamagui.config';
import {ChatModeBar} from '../ChatModeBar';
import type {ChatMode} from '../ChatModeBar';

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

jest.mock('../Icon', () => ({
  Icon: () => null,
}));

jest.mock('../../theme/colors', () => ({
  useColors: () => ({
    primary: '#7B52C4',
    primaryAlpha: () => 'rgba(123,82,196,0.15)',
    textMuted: '#888',
    background: '#fff',
    border: '#ddd',
  }),
}));

const Wrapper = ({children}: any) => (
  <TamaguiProvider config={appConfig} defaultTheme="light">{children}</TamaguiProvider>
);

const defaultProps = {
  mode: 'chat' as ChatMode,
  onModeChange: jest.fn(),
  tone: 'Friendly',
  onToneChange: jest.fn(),
  type: 'Email',
  onTypeChange: jest.fn(),
  rewriteStyle: 'Fix Grammar',
  onRewriteStyleChange: jest.fn(),
  decideStructure: 'Pros & Cons',
  onDecideStructureChange: jest.fn(),
  difficulty: 'Simple',
  onDifficultyChange: jest.fn(),
  langPair: 'EN→ES',
  onLangPairChange: jest.fn(),
  brainstormTopic: 'Name Ideas',
  onBrainstormTopicChange: jest.fn(),
  wellnessType: 'Sleep Story',
  onWellnessTypeChange: jest.fn(),
};

afterEach(cleanup);
beforeEach(() => jest.clearAllMocks());

async function renderBar(props: Partial<React.ComponentProps<typeof ChatModeBar>> = {}) {
  await act(async () => {
    render(<ChatModeBar {...defaultProps} {...props} />, {wrapper: Wrapper});
  });
}

describe('ChatModeBar', () => {
  it('renders without crashing', async () => {
    await renderBar();
    expect(screen.getByText('Chat')).toBeTruthy();
  });

  it('renders all mode labels', async () => {
    await renderBar();
    expect(screen.getByText('Chat')).toBeTruthy();
    expect(screen.getByText('Write')).toBeTruthy();
    expect(screen.getByText('Rewrite')).toBeTruthy();
    expect(screen.getByText('Ideas')).toBeTruthy();
    expect(screen.getByText('Wellness')).toBeTruthy();
  });

  it('calls onModeChange when a mode is pressed', async () => {
    await renderBar();
    await act(async () => {
      fireEvent.press(screen.getByText('Write'));
    });
    expect(defaultProps.onModeChange).toHaveBeenCalledWith('write');
  });

  it('shows write sub-options when in write mode', async () => {
    await renderBar({mode: 'write', tone: 'Professional', type: 'Email'});
    expect(screen.getByText('Friendly')).toBeTruthy();
    expect(screen.getByText('Professional')).toBeTruthy();
    expect(screen.getByText('Email')).toBeTruthy();
    expect(screen.getByText('Social Post')).toBeTruthy();
  });

  it('calls onToneChange when a tone is pressed', async () => {
    await renderBar({mode: 'write'});
    await act(async () => {
      fireEvent.press(screen.getByText('Professional'));
    });
    expect(defaultProps.onToneChange).toHaveBeenCalledWith('Professional');
  });

  it('shows rewrite sub-options when in rewrite mode', async () => {
    await renderBar({mode: 'rewrite'});
    expect(screen.getByText('Fix Grammar')).toBeTruthy();
    expect(screen.getByText('Make Shorter')).toBeTruthy();
    expect(screen.getByText('Make Professional')).toBeTruthy();
  });

  it('shows decide sub-options when in decide mode', async () => {
    await renderBar({mode: 'decide'});
    expect(screen.getByText('Pros & Cons')).toBeTruthy();
    expect(screen.getByText('Comparison')).toBeTruthy();
    expect(screen.getByText('Deep Analysis')).toBeTruthy();
  });

  it('shows explain sub-options when in explain mode', async () => {
    await renderBar({mode: 'explain'});
    expect(screen.getByText('Simple')).toBeTruthy();
    expect(screen.getByText('Moderate')).toBeTruthy();
    expect(screen.getByText('Expert')).toBeTruthy();
  });

  it('shows translate sub-options when in translate mode', async () => {
    await renderBar({mode: 'translate'});
    expect(screen.getByText('EN→ES')).toBeTruthy();
    expect(screen.getByText('EN→FR')).toBeTruthy();
    expect(screen.getByText('EN→DE')).toBeTruthy();
  });

  it('shows brainstorm sub-options when in brainstorm mode', async () => {
    await renderBar({mode: 'brainstorm'});
    expect(screen.getByText('Name Ideas')).toBeTruthy();
    expect(screen.getByText('Weekend Plans')).toBeTruthy();
    expect(screen.getByText('Gift Ideas')).toBeTruthy();
  });

  it('shows wellness sub-options when in wellness mode', async () => {
    await renderBar({mode: 'wellness'});
    expect(screen.getByText('Sleep Story')).toBeTruthy();
    expect(screen.getByText('Meditation')).toBeTruthy();
    expect(screen.getByText('Breathing')).toBeTruthy();
  });
});
