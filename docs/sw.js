const CACHE = 'crypto-bot-v9';

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  const path = new URL(e.request.url).pathname;
  // Nooit cachen: de HTML zelf en de bot-state
  if (path.endsWith('index.html') || path === '/' || path.endsWith('/')
      || path.endsWith('state.json')) return;
  // Cache-first voor overige assets (icons, manifest, sw.js)
  e.respondWith(
    caches.open(CACHE).then(cache =>
      cache.match(e.request).then(hit => {
        const net = fetch(e.request).then(r => { cache.put(e.request, r.clone()); return r; });
        return hit || net;
      })
    )
  );
});
