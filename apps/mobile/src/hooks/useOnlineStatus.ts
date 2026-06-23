import {useEffect, useState} from 'react';
import {getApiUrl} from '../services/api-client';

export function useOnlineStatus() {
  const [online, setOnline] = useState(true);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const url = await getApiUrl();
        const res = await fetch(url + '/health', {method: 'GET'});
        if (mounted) setOnline(res.ok);
      } catch {
        if (mounted) setOnline(false);
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
