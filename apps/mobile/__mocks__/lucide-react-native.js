const React = require('react');

const createMockIcon = name => {
  const MockIcon = React.forwardRef(({size, color, ...props}, ref) =>
    React.createElement('Icon', {testID: `lucide-${name}`, size, color, ref, ...props}),
  );
  MockIcon.displayName = `Lucide${name}`;
  return MockIcon;
};

const icons = {
  AArrowDown: createMockIcon('AArrowDown'),
  Copy: createMockIcon('Copy'),
  ExternalLink: createMockIcon('ExternalLink'),
  Reply: createMockIcon('Reply'),
  Forward: createMockIcon('Forward'),
  Pencil: createMockIcon('Pencil'),
  ArrowUp: createMockIcon('ArrowUp'),
  ArrowDown: createMockIcon('ArrowDown'),
  Pin: createMockIcon('Pin'),
  PinOff: createMockIcon('PinOff'),
  Star: createMockIcon('Star'),
  ThumbsUp: createMockIcon('ThumbsUp'),
  ThumbsDown: createMockIcon('ThumbsDown'),
  RefreshCw: createMockIcon('RefreshCw'),
  Trash2: createMockIcon('Trash2'),
  Check: createMockIcon('Check'),
  Mic: createMockIcon('Mic'),
  StopCircle: createMockIcon('StopCircle'),
  Zap: createMockIcon('Zap'),
  MessageCircle: createMockIcon('MessageCircle'),
  Brain: createMockIcon('Brain'),
  Dumbbell: createMockIcon('Dumbbell'),
  BookOpen: createMockIcon('BookOpen'),
  Search: createMockIcon('Search'),
  X: createMockIcon('X'),
  Menu: createMockIcon('Menu'),
  Settings: createMockIcon('Settings'),
  Sun: createMockIcon('Sun'),
  Moon: createMockIcon('Moon'),
  Package: createMockIcon('Package'),
  Archive: createMockIcon('Archive'),
  TriangleAlert: createMockIcon('TriangleAlert'),
  Palette: createMockIcon('Palette'),
  BarChart3: createMockIcon('BarChart3'),
  Plus: createMockIcon('Plus'),
  Bookmark: createMockIcon('Bookmark'),
  Share2: createMockIcon('Share2'),
  Image: createMockIcon('Image'),
  Upload: createMockIcon('Upload'),
  Download: createMockIcon('Download'),
  Lock: createMockIcon('Lock'),
  Unlock: createMockIcon('Unlock'),
  Info: createMockIcon('Info'),
  MoreHorizontal: createMockIcon('MoreHorizontal'),
  MoreVertical: createMockIcon('MoreVertical'),
  Music: createMockIcon('Music'),
  Headphones: createMockIcon('Headphones'),
  Target: createMockIcon('Target'),
  Save: createMockIcon('Save'),
  HelpCircle: createMockIcon('HelpCircle'),
  User: createMockIcon('User'),
  Book: createMockIcon('Book'),
  Square: createMockIcon('Square'),
  ArchiveX: createMockIcon('ArchiveX'),
  MessageSquare: createMockIcon('MessageSquare'),
  SmilePlus: createMockIcon('SmilePlus'),
};

icons.default = icons.Copy;

module.exports = icons;
module.exports.default = icons.Copy;
