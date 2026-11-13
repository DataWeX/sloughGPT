import {useErrorStore, addGlobalError} from '../error-store';

beforeEach(() => {
  useErrorStore.setState({errors: [], recentActivity: [], totalErrorCount: 0});
});

describe('error-store', () => {
  describe('addError', () => {
    it('adds an error from string', () => {
      const id = useErrorStore.getState().addError('Something failed');
      expect(id).toMatch(/^err_/);
      const {errors, totalErrorCount} = useErrorStore.getState();
      expect(errors).toHaveLength(1);
      expect(errors[0].message).toBe('Something failed');
      expect(errors[0].count).toBe(1);
      expect(totalErrorCount).toBe(1);
    });

    it('adds an error from Error object', () => {
      const id = useErrorStore.getState().addError(new TypeError('bad type'));
      expect(id).toMatch(/^err_/);
      const {errors} = useErrorStore.getState();
      expect(errors[0].message).toBe('bad type');
      expect(errors[0].title).toBe('Type Error');
    });

    it('adds an error from object with message', () => {
      const id = useErrorStore.getState().addError({message: 'timeout error'});
      expect(id).toMatch(/^err_/);
      const {errors} = useErrorStore.getState();
      expect(errors[0].message).toBe('timeout error');
    });

    it('adds an error from object with detail field', () => {
      useErrorStore.getState().addError({detail: 'rate limited'});
      const {errors} = useErrorStore.getState();
      expect(errors[0].message).toBe('rate limited');
    });

    it('adds an error from object with msg field', () => {
      useErrorStore.getState().addError({msg: 'bad request'});
      const {errors} = useErrorStore.getState();
      expect(errors[0].message).toBe('bad request');
    });

    it('handles non-serializable objects', () => {
      const id = useErrorStore.getState().addError({nested: {deep: true}});
      expect(id).toMatch(/^err_/);
    });

    it('deduplicates within 30s window', () => {
      const id1 = useErrorStore.getState().addError('timeout error', {source: 'api'});
      const id2 = useErrorStore.getState().addError('timeout error', {source: 'api'});
      expect(id1).toBe(id2);
      const {errors} = useErrorStore.getState();
      expect(errors).toHaveLength(1);
      expect(errors[0].count).toBe(2);
    });

    it('does not deduplicate different messages', () => {
      useErrorStore.getState().addError('error one', {source: 'api'});
      useErrorStore.getState().addError('error two', {source: 'api'});
      expect(useErrorStore.getState().errors).toHaveLength(2);
    });

    it('does not deduplicate different sources', () => {
      useErrorStore.getState().addError('timeout', {source: 'api'});
      useErrorStore.getState().addError('timeout', {source: 'web'});
      expect(useErrorStore.getState().errors).toHaveLength(2);
    });

    it('respects MAX_ERRORS limit (20)', () => {
      for (let i = 0; i < 25; i++) {
        useErrorStore.getState().addError(`error ${i}`);
      }
      expect(useErrorStore.getState().errors).toHaveLength(20);
    });

    it('uses explicit title when provided', () => {
      useErrorStore.getState().addError('msg', {title: 'Custom Title'});
      expect(useErrorStore.getState().errors[0].title).toBe('Custom Title');
    });

    it('uses explicit severity when provided', () => {
      useErrorStore.getState().addError('msg', {severity: 'info'});
      expect(useErrorStore.getState().errors[0].severity).toBe('info');
    });

    it('adds activity entry', () => {
      useErrorStore.getState().addError('Something failed');
      const {recentActivity} = useErrorStore.getState();
      expect(recentActivity).toHaveLength(1);
      expect(recentActivity[0].severity).toBe('error');
    });

    it('auto-detects severity from message', () => {
      useErrorStore.getState().addError('404 not found');
      expect(useErrorStore.getState().errors[0].severity).toBe('warning');

      useErrorStore.getState().addError('ECONNREFUSED');
      expect(useErrorStore.getState().errors[1].severity).toBe('warning');

      useErrorStore.getState().addError('timeout exceeded');
      expect(useErrorStore.getState().errors[2].severity).toBe('warning');
    });

    it('extracts title from 404 error', () => {
      useErrorStore.getState().addError({message: '404 Not Found'});
      expect(useErrorStore.getState().errors[0].title).toBe('Not Found');
    });

    it('extracts title from TypeError', () => {
      useErrorStore.getState().addError({name: 'TypeError', message: 'bad'});
      expect(useErrorStore.getState().errors[0].title).toBe('Error');
    });
  });

  describe('dismissError', () => {
    it('removes error by id', () => {
      const id = useErrorStore.getState().addError('fail');
      expect(useErrorStore.getState().errors).toHaveLength(1);
      useErrorStore.getState().dismissError(id);
      expect(useErrorStore.getState().errors).toHaveLength(0);
    });

    it('only removes the specified error', () => {
      const id1 = useErrorStore.getState().addError('error one', {source: 'a'});
      const id2 = useErrorStore.getState().addError('error two', {source: 'b'});
      useErrorStore.getState().dismissError(id1);
      expect(useErrorStore.getState().errors).toHaveLength(1);
      expect(useErrorStore.getState().errors[0].id).toBe(id2);
    });
  });

  describe('clearErrors', () => {
    it('clears all errors and activity', () => {
      useErrorStore.getState().addError('one');
      useErrorStore.getState().addError('two');
      expect(useErrorStore.getState().errors).toHaveLength(2);
      useErrorStore.getState().clearErrors();
      expect(useErrorStore.getState().errors).toHaveLength(0);
      expect(useErrorStore.getState().recentActivity).toHaveLength(0);
      expect(useErrorStore.getState().totalErrorCount).toBe(0);
    });
  });

  describe('getErrors', () => {
    it('returns current errors', () => {
      useErrorStore.getState().addError('fail');
      expect(useErrorStore.getState().getErrors()).toHaveLength(1);
    });
  });

  describe('hasErrors', () => {
    it('returns true when errors exist', () => {
      useErrorStore.getState().addError('fail');
      expect(useErrorStore.getState().hasErrors()).toBe(true);
    });

    it('returns false when no errors', () => {
      expect(useErrorStore.getState().hasErrors()).toBe(false);
    });
  });

  describe('addGlobalError', () => {
    it('adds error via global function', () => {
      const id = addGlobalError('global fail', 'test');
      expect(id).toMatch(/^err_/);
      expect(useErrorStore.getState().errors[0].source).toBe('test');
    });
  });
});
