/**
 * Tests for sensor collection service.
 */

import {isCollecting, getCollectionStats} from '../sensor-collection';

describe('sensor-collection', () => {
  it('isCollecting returns false initially', () => {
    expect(isCollecting()).toBe(false);
  });

  it('getCollectionStats returns default stats', () => {
    const stats = getCollectionStats();
    expect(stats.totalReadings).toBe(0);
    expect(stats.totalWindows).toBe(0);
    expect(stats.lastFlushTime).toBeNull();
    expect(stats.isCollecting).toBe(false);
  });
});
