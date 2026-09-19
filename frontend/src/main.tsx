import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

// Wake-up ping: fires the moment the page loads, before React even mounts.
// On a serverless backend this starts the cold boot while the user is
// still reading the homescreen. Aborted/failed pings still trigger the
// wake server-side. Goes through the same-origin proxy in dev and prod.
fetch('/healthz', { cache: 'no-store' }).catch(() => {});

createRoot(document.getElementById('root')!).render(<App />);