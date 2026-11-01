import React from 'react';
import {render} from '@testing-library/react-native';
import {Icon, IconName} from '../Icon';

jest.mock('lucide-react-native', () => {
  const React = require('react');
  const {View} = require('react-native');
  const MockIcon = ({size, color}: {size?: number; color?: string}) =>
    React.createElement(View, {style: {width: size, height: size, backgroundColor: color}});
  return {
    Copy: MockIcon,
    ExternalLink: MockIcon,
    Reply: MockIcon,
    Forward: MockIcon,
    Pencil: MockIcon,
    ArrowUp: MockIcon,
    ArrowDown: MockIcon,
    Pin: MockIcon,
    PinOff: MockIcon,
    Star: MockIcon,
    ThumbsUp: MockIcon,
    ThumbsDown: MockIcon,
    RefreshCw: MockIcon,
    Trash2: MockIcon,
    Check: MockIcon,
    Mic: MockIcon,
    StopCircle: MockIcon,
    Zap: MockIcon,
    MessageCircle: MockIcon,
    Brain: MockIcon,
    Dumbbell: MockIcon,
    BookOpen: MockIcon,
    Search: MockIcon,
    X: MockIcon,
    Menu: MockIcon,
    Settings: MockIcon,
    Sun: MockIcon,
    Moon: MockIcon,
    Package: MockIcon,
    Archive: MockIcon,
    TriangleAlert: MockIcon,
    Palette: MockIcon,
    BarChart3: MockIcon,
    Plus: MockIcon,
    Bookmark: MockIcon,
    Share2: MockIcon,
    Image: MockIcon,
    Upload: MockIcon,
    Download: MockIcon,
    Lock: MockIcon,
    Unlock: MockIcon,
    Info: MockIcon,
    MoreHorizontal: MockIcon,
    MoreVertical: MockIcon,
    Music: MockIcon,
    Headphones: MockIcon,
    Target: MockIcon,
    Save: MockIcon,
    HelpCircle: MockIcon,
    User: MockIcon,
    Book: MockIcon,
    Square: MockIcon,
    ArchiveX: MockIcon,
    MessageSquare: MockIcon,
    SmilePlus: MockIcon,
  };
});

const ALL_ICONS: IconName[] = [
  'copy', 'external-link', 'reply', 'forward', 'edit',
  'pin', 'pin-off', 'star', 'star-outline',
  'smile-plus', 'thumbs-up', 'thumbs-down',
  'refresh-cw', 'trash-2', 'check', 'mic',
  'arrow-up', 'arrow-down',
  'stop-circle', 'zap', 'message-circle',
  'brain', 'dumbbell', 'book-open',
  'search', 'x', 'menu', 'settings',
  'sun', 'moon', 'package', 'archive',
  'triangle-alert', 'palette', 'bar-chart', 'plus',
  'bookmark', 'share-2', 'image', 'upload',
  'download', 'lock', 'unlock', 'info',
  'more-horizontal', 'more-vertical',
  'music', 'headphones', 'target', 'save',
  'help', 'user', 'book', 'square',
  'archive-x', 'message-square',
];

describe('Icon', () => {
  it('renders without crashing with default props', () => {
    expect(() => render(<Icon name="star" />)).not.toThrow();
  });

  it('renders with custom size', () => {
    expect(() => render(<Icon name="star" size={32} />)).not.toThrow();
  });

  it('renders with custom color', () => {
    expect(() => render(<Icon name="star" color="red" />)).not.toThrow();
  });

  it('renders nothing for unknown icon name', () => {
    // @ts-expect-error — testing runtime fallback
    expect(() => render(<Icon name="nonexistent" />)).not.toThrow();
  });

  it.each(ALL_ICONS)('renders icon "%s" without crashing', name => {
    expect(() => render(<Icon name={name} />)).not.toThrow();
  });
});
