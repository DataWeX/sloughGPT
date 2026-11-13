import {useEffect, useState, useRef} from 'react';
import {getApiUrl} from '../services/api-client';

/**
 * Tracks server connectivity. Returns `false` only after the server was
 * previously reachable and then went down. If the server was never reached
 * (e.g. first launch with no backend), returns `true` — avoids showing
 * the offline banner on every cold start.
 */
export function useOnlineStatus() {
  const [online, setOnline] = useState(true);
  const everConnected = useRef(false);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const url = await getApiUrl();
        const res = await fetch(url + '/health', {method: 'GET'});
        if (!mounted) return;
        if (res.ok) {
          everConnected.current = true;
          setOnline(true);
        } else if (everConnected.current) {
          setOnline(false);
        }
      } catch {
        if (!mounted) return;
        if (everConnected.current) {
          setOnline(false);
        }
      }
    };
    check();
    const interval = setInterval(check, 10000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return online;
}
