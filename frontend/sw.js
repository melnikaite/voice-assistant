// Service worker — Web Push only (no static cache, no offline shell).
//
// Lifecycle on the browser side:
//   1. main.js calls navigator.serviceWorker.register('/sw.js') after a
//      successful login.  The SW installs and activates immediately
//      (we skipWaiting + claim so the first push works without a reload).
//   2. main.js calls reg.pushManager.subscribe(...) with the VAPID public
//      key fetched from /api/push/vapid_public_key, then POSTs the
//      subscription JSON to /api/push/subscribe so the orchestrator can
//      reach the user when every tab is closed.
//   3. When the orchestrator pushes a voicemail event, the push service
//      wakes this worker — even if there's no tab open — and we surface
//      a system notification.  Clicking it focuses an existing tab or
//      opens the app rooted at the voicemail.
//
// Out of scope for this sprint: caching static assets, PWA installability,
// offline fallback page.  Push is purely additive over the WS path.

self.addEventListener('install', (event) => {
  // skipWaiting() lets a fresh install activate without waiting for all
  // existing pages to close — the first push registration will land on
  // the new worker the moment the user clicks "subscribe".
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  // clients.claim() means already-open tabs get controlled by this
  // worker immediately, so a push that arrives while a tab is open is
  // still routed through showNotification (consistent with the closed
  // case).  Without claim, the first SW only takes effect on next reload.
  event.waitUntil(self.clients.claim());
});

self.addEventListener('push', (event) => {
  // Push payload shapes:
  //   voicemail:  {title, body, voicemail_id, tag}
  //   step-up:    {type:"step_up_request", title, body, tag, token, open_url}
  // event.data may be null on test/empty pushes — fall back to the
  // generic shape so the user still sees something.
  let payload = {};
  if (event.data) {
    try {
      payload = event.data.json();
    } catch (e) {
      payload = { title: 'Voice Assistant', body: event.data.text() };
    }
  }
  const title = payload.title || 'Voice Assistant';
  const isStepUp = payload.type === 'step_up_request';
  const options = {
    body: payload.body || '',
    tag: payload.tag || (payload.voicemail_id ? `voicemail-${payload.voicemail_id}` : undefined),
    renotify: false,
    data: {
      voicemail_id: payload.voicemail_id ?? null,
      // Step-up: carry the opaque grant token; the click handler POSTs it
      // to /api/step-up/approve (token never goes in a URL).
      step_up_token: isStepUp ? (payload.token || null) : null,
      // Non-step-up: navigate to this URL on tap.
      open_url: !isStepUp
        ? (payload.voicemail_id ? `/?voicemail=${payload.voicemail_id}` : '/')
        : null,
    },
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', (event) => {
  // Two kinds of notification clicks:
  //
  //   Step-up approval  (data.step_up_approve_url is set):
  //     Do a background fetch to POST /api/step-up/approve with the
  //     token embedded in the URL.  The orchestrator broadcasts
  //     step_up_granted over WS; the open tab receives it and sets the
  //     session flag so the next voice turn can execute the gated tool.
  //     We focus an existing tab so the user sees the confirmation
  //     toast, but don't navigate — the tap itself is the auth action.
  //
  //   Voicemail / generic  (data.open_url is set):
  //     Focus an existing same-origin tab or open a new one, then
  //     postMessage so main.js can jump to the linked content.
  event.notification.close();
  const notifData = event.notification.data || {};
  const stepUpToken = notifData.step_up_token;
  const openUrl = notifData.open_url || '/';

  event.waitUntil(
    (async () => {
      // ── Step-up path ────────────────────────────────────────────────
      if (stepUpToken) {
        try {
          // Background fetch — no tab required.  POST the opaque grant
          // token in the body (never a URL/query string) so it can't
          // leak via access logs / Referer / history.
          await fetch('/api/step-up/approve', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token: stepUpToken }),
          });
        } catch (e) {
          // Best-effort — if the tab is open it will show an error toast.
        }
        // Focus the open tab so the user sees the step_up_granted WS
        // event's confirmation toast ("Confirmed! Ask again.").
        const allClients = await self.clients.matchAll({
          type: 'window',
          includeUncontrolled: true,
        });
        for (const client of allClients) {
          try {
            if (new URL(client.url).origin === self.location.origin) {
              await client.focus();
              return;
            }
          } catch {}
        }
        return; // No open tab — approval was still sent in background.
      }

      // ── Voicemail / generic path ─────────────────────────────────────
      const allClients = await self.clients.matchAll({
        type: 'window',
        includeUncontrolled: true,
      });
      // Reuse any same-origin tab — focus it, then ask main.js to jump
      // to the linked voicemail.  Tabs on a different origin (e.g. an
      // iframe embed) are ignored.
      for (const client of allClients) {
        try {
          const url = new URL(client.url);
          if (url.origin === self.location.origin) {
            await client.focus();
            try {
              client.postMessage({
                type: 'sw_notification_click',
                voicemail_id: notifData.voicemail_id || null,
                open_url: openUrl,
              });
            } catch (e) {
              // postMessage can fail if the tab is mid-navigation;
              // not fatal — the focus already happened.
            }
            return;
          }
        } catch (e) {
          // Malformed client.url — ignore and try the next one.
        }
      }
      // No existing tab — open a fresh one.  ``clients.openWindow`` may
      // return null if the browser blocked it (no user gesture), but
      // at the OS-notification level the click IS a gesture so this is
      // expected to succeed.
      await self.clients.openWindow(openUrl);
    })(),
  );
});

self.addEventListener('pushsubscriptionchange', (event) => {
  // Some push services force-rotate the subscription endpoint without
  // the user touching anything.  When that happens we MUST grab the
  // new subscription and re-POST it to /api/push/subscribe; otherwise
  // the orchestrator keeps pushing to the dead old endpoint, gets 410,
  // and GCs the row.
  //
  // We deliberately keep this minimal — re-subscribe with the same
  // applicationServerKey we already have (fetched fresh from the
  // server so we don't have to bake it into SW state).
  event.waitUntil(
    (async () => {
      try {
        const r = await fetch('/api/push/vapid_public_key');
        if (!r.ok) return;
        const data = await r.json();
        const key = _b64UrlToUint8Array(data.public_key);
        const sub = await self.registration.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: key,
        });
        await fetch('/api/push/subscribe', {
          method: 'POST',
          credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(sub.toJSON()),
        });
      } catch (e) {
        // Best-effort — next page load triggers a fresh subscribe.
      }
    })(),
  );
});

// Tiny VAPID-key decoder copy.  We can't import from main.js inside a
// service worker (different module scope), so this stays in sync by
// being a verbatim mirror of the helper in main.js.  Keep them
// identical if you ever touch one of them.
function _b64UrlToUint8Array(b64Url) {
  const padding = '='.repeat((4 - (b64Url.length % 4)) % 4);
  const b64 = (b64Url + padding).replace(/-/g, '+').replace(/_/g, '/');
  const raw = atob(b64);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}
