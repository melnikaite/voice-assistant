// Frontend i18n — mirrors orchestrator/app/i18n.py.
//
// User-facing strings live here keyed by stable English identifiers;
// values per language.  Code paths call `t("key")` (which reads the
// currently-selected language) or `t("key", lang)` for an explicit
// override.  Apply at boot via `applyI18n()` which walks every element
// with a `data-i18n` attribute and replaces its textContent — keeps
// index.html readable (English-by-default-but-keyed) and avoids
// scattered `if (lang === 'ru')` branches across main.js.
//
// Adding a string:
//   1. Pick a stable key (dot-namespaced, e.g. "settings.save").
//   2. Add an entry to CATALOG with en/ru/de values.
//   3. In HTML: <button data-i18n="settings.save">Save</button>
//   4. In JS:  someEl.textContent = t("settings.save");
//
// Language source of truth:
//   • localStorage key `va_lang` — set by the Settings panel.
//   • If unset, falls back to `navigator.language`'s first two
//     letters, then to "en".
// On login, the auth flow refreshes from the user's settings.language
// (auto / en / ru / de) — "auto" → don't pin, keep the
// localStorage/navigator value.

export const SUPPORTED_LANGS = ['en', 'ru', 'de'];
export const DEFAULT_LANG = 'en';

export const CATALOG = {
  // ── Page chrome ──────────────────────────────────────────────
  'app.title':       { en: 'Voice Assistant',       ru: 'Голосовой ассистент',     de: 'Sprachassistent' },
  'app.tagline':     {
    en: 'Say «hey jarvis», then a command.  WebRTC + server-side TTS — you can interrupt by speaking.',
    ru: 'Скажи «hey jarvis», потом команду.  WebRTC + server-side TTS — можно перебивать голосом.',
    de: 'Sag «hey jarvis», dann den Befehl.  WebRTC + Server-TTS — du kannst sprechend unterbrechen.',
  },

  // ── Controls ─────────────────────────────────────────────────
  'controls.start':  { en: 'Start',   ru: 'Старт',  de: 'Start' },
  'controls.stop':   { en: 'Stop',    ru: 'Стоп',   de: 'Stopp' },
  'controls.cancel': { en: 'Cancel',  ru: 'Отмена', de: 'Abbrechen' },
  'controls.ptt':    { en: '🎤 Talk', ru: '🎤 Говорить', de: '🎤 Sprechen' },

  // ── State card ───────────────────────────────────────────────
  'state.idle':        { en: 'inactive',           ru: 'не активен',                de: 'inaktiv' },
  'state.idle_hint':   {
    en: 'Press «Start» to enable the microphone',
    ru: 'Нажми «Старт» чтобы включить микрофон',
    de: 'Drücke «Start», um das Mikrofon zu aktivieren',
  },
  'state.mute_sounds': { en: 'mute sounds', ru: 'без звуков', de: 'stumm' },

  // ── Current request panel ────────────────────────────────────
  'current.header':   { en: 'Current request', ru: 'Текущий запрос', de: 'Aktuelle Anfrage' },
  'current.heard':    { en: 'Heard:',  ru: 'Услышал:',  de: 'Gehört:' },
  'current.tool':     { en: 'Tool:',   ru: 'Tool:',     de: 'Tool:' },
  'current.response': { en: 'Reply:',  ru: 'Ответ:',    de: 'Antwort:' },

  // ── Auth ─────────────────────────────────────────────────────
  'auth.signin':         { en: 'Sign in',  ru: 'Вход',  de: 'Anmelden' },
  'auth.profile':        { en: 'Profile',  ru: 'Профиль', de: 'Profil' },
  'auth.pick_profile':   { en: '— pick a profile —', ru: '— выбери профиль —', de: '— Profil wählen —' },
  'auth.password':       { en: 'Password', ru: 'Пароль', de: 'Passwort' },
  'auth.login':          { en: 'Log in',   ru: 'Войти',  de: 'Einloggen' },
  'auth.logout':         { en: 'Log out',  ru: 'Выйти',  de: 'Ausloggen' },
  'auth.change_pw':      { en: 'Change password', ru: 'Сменить пароль', de: 'Passwort ändern' },
  'auth.signed_in_as':   { en: 'Signed in as',  ru: 'Вошёл как',  de: 'Angemeldet als' },
  'auth.valid_until':    { en: 'valid until', ru: 'действует до', de: 'gültig bis' },
  'auth.first_run':      { en: 'First run', ru: 'Первый запуск', de: 'Erster Start' },
  'auth.first_run_hint': {
    en: 'Create a password for your profile — needed to confirm deferred actions and edit memory.',
    ru: 'Создай пароль для своего профиля — он понадобится для подтверждения отложенных действий и редактирования памяти.',
    de: 'Lege ein Passwort für dein Profil fest — wird für aufgeschobene Aktionen und Speicheränderungen benötigt.',
  },
  'auth.new_password':   { en: 'New password', ru: 'Новый пароль', de: 'Neues Passwort' },
  'auth.create':         { en: 'Create', ru: 'Создать', de: 'Erstellen' },

  // ── Pending queue ─────────────────────────────────────────────
  'pending.header':    { en: 'Deferred actions', ru: 'Отложенные действия', de: 'Aufgeschobene Aktionen' },
  'pending.hint':      {
    en: 'Actions the assistant decided to park (no password or explicit "later").  They expire in 24 h.',
    ru: 'Действия, которые ассистент решил отложить (из-за отсутствия пароля или явной просьбы «не сейчас»).  Истекают через 24 ч.',
    de: 'Aktionen, die der Assistent geparkt hat (kein Passwort oder „später").  Verfallen nach 24 h.',
  },
  'pending.refresh':   { en: 'Refresh', ru: 'Обновить', de: 'Aktualisieren' },
  'pending.approve':   { en: 'Approve', ru: 'Одобрить', de: 'Genehmigen' },
  'pending.reject':    { en: 'Reject',  ru: 'Отклонить', de: 'Ablehnen' },
  'pending.none':      { en: 'Nothing parked.', ru: 'Ничего не отложено.', de: 'Nichts geparkt.' },
  'pending.loading':   { en: 'Loading…', ru: 'Загружаю…', de: 'Lädt…' },
  'pending.recent':    { en: 'Recent:', ru: 'Недавно:', de: 'Kürzlich:' },

  // ── Settings ──────────────────────────────────────────────────
  'settings.header':   { en: 'Settings', ru: 'Настройки', de: 'Einstellungen' },
  'settings.language': { en: 'Language', ru: 'Язык', de: 'Sprache' },
  'settings.tone':     { en: 'Tone',     ru: 'Тон',  de: 'Ton' },
  'settings.voice':    { en: 'Voice (XTTS speaker name or clone:<id>)', ru: 'Голос (XTTS speaker name или clone:<id>)', de: 'Stimme (XTTS-Sprecher oder clone:<id>)' },
  'settings.save':     { en: 'Save', ru: 'Сохранить', de: 'Speichern' },
  'settings.reload':   { en: 'Reload', ru: 'Перечитать', de: 'Neu laden' },

  // ── Memory ────────────────────────────────────────────────────
  'memory.header':     { en: 'Memory', ru: 'Память', de: 'Speicher' },
  'memory.hint':       {
    en: 'Freeform notes about you — the assistant reads them before every reply and writes here on «remember …» phrases.  Markdown, manually editable.',
    ru: 'Свободные заметки про тебя — ассистент читает их перед каждым ответом и пишет сюда по фразам «запомни …».  Markdown, редактируется вручную.',
    de: 'Freie Notizen über dich — der Assistent liest sie vor jeder Antwort und schreibt hierhin bei «merk dir …».  Markdown, manuell bearbeitbar.',
  },

  // ── Stats ─────────────────────────────────────────────────────
  'stats.header': { en: 'Stats', ru: 'Статистика', de: 'Statistik' },
  'stats.period': { en: 'Period:', ru: 'Период:', de: 'Zeitraum:' },
  'stats.day':    { en: 'Day',   ru: 'Сутки',   de: 'Tag' },
  'stats.week':   { en: 'Week',  ru: 'Неделя',  de: 'Woche' },
  'stats.month':  { en: 'Month', ru: 'Месяц',   de: 'Monat' },

  // ── Log panel ────────────────────────────────────────────────
  'log.header':  { en: 'Log', ru: 'Лог', de: 'Log' },

  // ── Inbox (voicemail panel) ──────────────────────────────────
  'inbox.title': { en: 'Voicemail', ru: 'Сообщения', de: 'Sprachnachrichten' },
  'inbox.help':  {
    en: 'Messages other speakers left for you via «передай X что …» / «leave a message for X: …».  Click a row to play the original recording — that\'s the actual sender\'s voice.',
    ru: 'Сообщения, которые другие оставили тебе через «передай X что …».  Нажми на строку чтобы прослушать оригинал — голос отправителя.',
    de: 'Nachrichten, die andere für dich hinterlassen haben mit «hinterlass eine Nachricht für …».  Klick eine Zeile an, um die Originalaufnahme abzuspielen.',
  },
  'inbox.loading':     { en: 'Loading…', ru: 'Загружаю…', de: 'Lädt…' },
  'inbox.empty':       { en: 'No voicemail.', ru: 'Сообщений нет.', de: 'Keine Nachrichten.' },
  'inbox.refresh':     { en: 'Refresh', ru: 'Обновить', de: 'Aktualisieren' },
  'inbox.unread_only': { en: 'Unread only', ru: 'Только новые', de: 'Nur ungelesen' },
  'inbox.play':        { en: 'Play', ru: 'Воспроизвести', de: 'Abspielen' },
  'inbox.reply':       { en: 'Reply', ru: 'Ответить', de: 'Antworten' },
  'inbox.reply_placeholder': {
    en: 'Type your reply…', ru: 'Текст ответа…', de: 'Antwort eingeben…',
  },
  'inbox.send':        { en: 'Send', ru: 'Отправить', de: 'Senden' },
  'inbox.delete':      { en: 'Delete', ru: 'Удалить', de: 'Löschen' },
  'inbox.from':        { en: 'from', ru: 'от', de: 'von' },
  'inbox.guest':       { en: 'guest', ru: 'гость', de: 'Gast' },
  // Inbox / Sent toggle.  Default view is Inbox; switching to Sent
  // queries /outgoing_voicemail and renders rows from the sender's
  // POV (no reply input, no delete — sender owns the send, not the
  // mutation rights on the recipient's row).
  'inbox.mode_label':  { en: 'View',  ru: 'Режим',  de: 'Ansicht' },
  'inbox.mode_inbox':  { en: 'Inbox', ru: 'Входящие', de: 'Posteingang' },
  'inbox.mode_sent':   { en: 'Sent',  ru: 'Отправленные', de: 'Gesendet' },
  'inbox.to':          { en: 'to',    ru: 'для',    de: 'an' },
  'inbox.no_reply_yet': {
    en: 'no reply yet',
    ru: 'ответа пока нет',
    de: 'noch keine Antwort',
  },
  'inbox.outgoing_empty': {
    en: 'No sent messages.',
    ru: 'Отправленных сообщений нет.',
    de: 'Keine gesendeten Nachrichten.',
  },
  // Desktop notification (when the tab is hidden and the user
  // pre-granted permission).  ``from_name`` is substituted on the
  // JS side — Notification API doesn't run str.format.
  'inbox.notify_title': {
    en: 'New voicemail',
    ru: 'Новое сообщение',
    de: 'Neue Nachricht',
  },
  'inbox.notify_body': {
    en: '{from_name} left you a message',
    ru: '{from_name} оставил сообщение',
    de: '{from_name} hat eine Nachricht hinterlassen',
  },

  // ── State card labels ────────────────────────────────────────────
  'state.listening_wake.label': { en: 'Waiting for wake word',        ru: 'Жду пробуждения',         de: 'Warte auf Aktivierungswort' },
  'state.listening_wake.sub':   { en: 'Say «hey jarvis» or press «Talk»', ru: 'Скажи «hey jarvis» или нажми «Говорить»', de: 'Sag «hey jarvis» oder drücke «Sprechen»' },
  'state.recording.label':      { en: 'Listening',                    ru: 'Слушаю',                  de: 'Höre zu' },
  'state.recording.sub':        { en: 'Recording your command…',      ru: 'Запись твоей команды…',   de: 'Nimm deinen Befehl auf…' },
  'state.processing.label':     { en: 'Thinking',                     ru: 'Думаю',                   de: 'Denke nach' },
  'state.processing.sub':       { en: 'Hold Space to add more.',      ru: 'Зажми Space, чтобы дополнить.', de: 'Leertaste halten, um mehr hinzuzufügen.' },
  'state.speaking.label':       { en: 'Speaking',                     ru: 'Говорю',                  de: 'Spreche' },
  'state.speaking.sub':         { en: 'You can interrupt — just start talking', ru: 'Можешь перебить — просто начни говорить', de: 'Du kannst unterbrechen — fang einfach an zu sprechen' },
  'state.continuation.label':   { en: 'Listening for follow-up',      ru: 'Слушаю продолжение',      de: 'Warte auf Fortsetzung' },
  'state.continuation.sub':     { en: 'You can ask more without the wake word', ru: 'Можешь спросить ещё без wake-слова', de: 'Du kannst ohne Aktivierungswort weiterfragen' },
  'state.loading_models.label': { en: 'Loading models',               ru: 'Загружаю модели',         de: 'Lade Modelle' },
  'state.loading_models.sub':   { en: 'Once at startup, then cached', ru: 'Один раз при старте, дальше из кэша', de: 'Einmalig beim Start, dann aus dem Cache' },

  // ── Progress stage labels ────────────────────────────────────────
  'progress.asr':              { en: 'Recognizing speech',             ru: 'Распознаю речь',           de: 'Erkenne Sprache' },
  'progress.agent':            { en: 'Choosing tool',                  ru: 'Выбираю инструмент',       de: 'Wähle Werkzeug' },
  'progress.tool':             { en: 'Running',                        ru: 'Запускаю',                 de: 'Starte' },
  'progress.localize':         { en: 'Translating query for search',   ru: 'Перевожу запрос для поиска', de: 'Übersetze Anfrage für Suche' },
  'progress.search':           { en: 'Searching the web',              ru: 'Ищу в интернете',          de: 'Suche im Internet' },
  'progress.fetch':            { en: 'Reading pages',                  ru: 'Читаю страницы',           de: 'Lese Seiten' },
  'progress.summarize':        { en: 'Composing reply',                ru: 'Формирую ответ',           de: 'Formuliere Antwort' },
  'progress.thinking':         { en: 'Thinking through the answer',    ru: 'Думаю над ответом',        de: 'Denke nach über die Antwort' },
  'progress.create_reminder':  { en: 'Setting reminder',               ru: 'Ставлю напоминание',       de: 'Erstelle Erinnerung' },
  'progress.cancel_reminder':  { en: 'Cancelling reminder',            ru: 'Отменяю напоминание',      de: 'Absage Erinnerung' },
  'progress.list_reminders':   { en: 'Checking my reminders',          ru: 'Смотрю мои напоминания',   de: 'Prüfe meine Erinnerungen' },

  // ── Continuation countdown ───────────────────────────────────────
  // {n} = seconds remaining
  'state.continuation.countdown': { en: '(listening {n}s)', ru: '(слушаю {n}с)', de: '(höre {n}s zu)' },

  // ── Transcript / tool display ────────────────────────────────────
  'transcript.empty':          { en: '(empty)',     ru: '(пусто)',    de: '(leer)' },
  'tool.none':                 { en: '(no tool)',   ru: '(нет tool)', de: '(kein Tool)' },

  // ── Push TTS prefix ──────────────────────────────────────────────
  'push_tts.missed_reminder_prefix': { en: 'Missed reminder: ', ru: 'Пропущенное напоминание: ', de: 'Verpasste Erinnerung: ' },

  // ── Attach image preview ─────────────────────────────────────────
  // {name} = filename, {kb} = size in KB
  'attach.meta': { en: '{name} · {kb} KB · will send with next utterance', ru: '{name} · {kb} KB · отправится со следующей репликой', de: '{name} · {kb} KB · wird mit nächster Äußerung gesendet' },

  // ── Speaker enrollment panel ─────────────────────────────────────
  'speaker.no_profiles':       { en: 'No profiles',            ru: 'Нет профилей',             de: 'Keine Profile' },
  'speaker.voice_default':     { en: '— default —',            ru: '— по умолчанию —',         de: '— Standard —' },
  'speaker.optgroup_custom':   { en: 'Custom',                 ru: 'Кастомные',                de: 'Eigene' },
  'speaker.optgroup_builtin':  { en: 'Built-in',               ru: 'Встроенные',               de: 'Eingebaut' },
  // {name} = speaker name, {voice} = voice name
  'speaker.voice_changed':     { en: '"{name}" → voice {voice}', ru: '"{name}" → голос {voice}', de: '"{name}" → Stimme {voice}' },
  'speaker.voice_default_log': { en: 'default',                ru: 'по умолчанию',             de: 'Standard' },
  'speaker.voice_update_fail': { en: 'failed to update voice ({err})', ru: 'не удалось обновить голос ({err})', de: 'Stimme konnte nicht aktualisiert werden ({err})' },
  'speaker.deleted':           { en: 'deleted profile "{name}"', ru: 'удалён профиль "{name}"', de: 'Profil "{name}" gelöscht' },
  'speaker.recording':         { en: '🔴 Recording… {s}s — speak!', ru: '🔴 Записываю… {s} с — говори!', de: '🔴 Aufnahme… {s}s — sprich!' },
  'speaker.uploading':         { en: 'Uploading to server…',   ru: 'Отправляю на сервер…',     de: 'Lade auf Server hoch…' },
  // {name} = profile name
  'speaker.saved':             { en: '✅ Profile «{name}» saved!', ru: '✅ Профиль «{name}» сохранён!', de: '✅ Profil «{name}» gespeichert!' },
  'speaker.error':             { en: '❌ Error: {err}',         ru: '❌ Ошибка: {err}',         de: '❌ Fehler: {err}' },
  'speaker.name_required':     { en: 'Enter a name before recording', ru: 'Введи имя перед записью', de: 'Gib einen Namen ein vor der Aufnahme' },

  // ── Custom voice (clone) panel ────────────────────────────────────
  'voice.no_clones':           { en: 'None yet — record your first', ru: 'Пока нет — запиши первый', de: 'Noch keine — nimm die erste auf' },
  'voice.recording':           { en: '🔴 Recording… {s}s — speak naturally, varied intonation', ru: '🔴 Записываю… {s} с — говори естественно, разными интонациями', de: '🔴 Aufnahme… {s}s — sprich natürlich mit verschiedenen Intonationen' },
  'voice.uploading':           { en: 'Uploading to server…',   ru: 'Отправляю на сервер…',     de: 'Lade auf Server hoch…' },
  'voice.saved':               { en: '✅ Voice «{name}» saved!', ru: '✅ Голос «{name}» сохранён!', de: '✅ Stimme «{name}» gespeichert!' },
  'voice.error':               { en: '❌ Error: {err}',         ru: '❌ Ошибка: {err}',         de: '❌ Fehler: {err}' },
  'voice.deleted':             { en: 'deleted «{name}»',       ru: 'удалён «{name}»',          de: '«{name}» gelöscht' },
  'voice.name_required':       { en: 'Enter a name before recording', ru: 'Введи название перед записью', de: 'Gib einen Namen ein vor der Aufnahme' },
  'voice.load_error':          { en: 'Failed to load voice list: {err}', ru: 'Не удалось загрузить список голосов: {err}', de: 'Stimmenliste konnte nicht geladen werden: {err}' },

  // ── Stats panel ──────────────────────────────────────────────────
  'stats.loading':             { en: 'Loading…',              ru: 'Загружаю…',                de: 'Lädt…' },
  'stats.updated':             { en: 'Updated: {time}',       ru: 'Обновлено: {time}',        de: 'Aktualisiert: {time}' },
  'stats.error':               { en: 'Error: {err}',          ru: 'Ошибка: {err}',            de: 'Fehler: {err}' },
  'stats.no_data':             { en: 'No data for selected period.', ru: 'Пока нет данных за выбранный период.', de: 'Keine Daten für den gewählten Zeitraum.' },
  // {total}, {prompt}, {completion} = formatted token counts
  'stats.tokens_summary':      {
    en: 'Total tokens: {total} (prompt {prompt}, completion {completion}), local Gemma — free.',
    ru: 'Всего токенов: {total} (prompt {prompt}, completion {completion}), локальный Gemma — бесплатно.',
    de: 'Token gesamt: {total} (Prompt {prompt}, Completion {completion}), lokales Gemma — kostenlos.',
  },
  // {usd_claude}, {usd_mini} = formatted dollar amounts
  'stats.cost_if':             {
    en: 'If routed through Claude Sonnet 4.5: {usd_claude}. If through GPT-4o-mini: {usd_mini}.',
    ru: 'Если бы это шло через Claude Sonnet 4.5: {usd_claude}. Если бы через GPT-4o-mini: {usd_mini}.',
    de: 'Über Claude Sonnet 4.5: {usd_claude}. Über GPT-4o-mini: {usd_mini}.',
  },
  'stats.chartjs_unavailable': { en: 'Chart.js not loaded — charts unavailable.', ru: 'Chart.js не загрузился — графики недоступны.', de: 'Chart.js nicht geladen — Diagramme nicht verfügbar.' },
  // {n} = count of remaining clients in per-user chart
  'stats.more_clients':        { en: '{n} more clients', ru: 'ещё клиентов ({n})', de: 'noch {n} Kunden' },
  // Step-up auth UI
  'step_up.granted': {
    en: '✓ Confirmed ({min} min window). Ask again.',
    ru: '✓ Подтверждено (окно {min} мин). Повтори запрос.',
    de: '✓ Bestätigt ({min} Min. Fenster). Frag erneut.',
  },

  // System health strip
  'stats.sys_sessions':        { en: 'Active sessions', ru: 'Сессий', de: 'Sitzungen' },
  'stats.sys_agents':          { en: 'Agents',          ru: 'Агентов', de: 'Agenten' },
  'stats.sys_uptime':          { en: 'Uptime',          ru: 'Аптайм',  de: 'Laufzeit' },
  'stats.sys_turns':           { en: 'Turns today',     ru: 'Запросов за день', de: 'Anfragen heute' },
  // Tool performance table headers
  'stats.label_tool_perf':     { en: 'Tool performance',  ru: 'Производительность инструментов', de: 'Tool-Performance' },
  'stats.perf_tool':           { en: 'Tool',     ru: 'Инструмент', de: 'Tool'   },
  'stats.perf_calls':          { en: 'Calls',    ru: 'Вызовов',    de: 'Aufrufe' },
  'stats.perf_avg_ms':         { en: 'Avg ms',   ru: 'Сред. мс',   de: 'Ø ms'   },
  'stats.perf_errors':         { en: 'Errors',   ru: 'Ошибок',     de: 'Fehler' },

  // ── Auth status messages ─────────────────────────────────────────
  'auth.profile_number':       { en: 'profile #{id}',          ru: 'профиль #{id}',            de: 'Profil #{id}' },
  'auth.select_profile_and_password': { en: 'Select a profile and enter password.', ru: 'Выбери профиль и введи пароль.', de: 'Profil auswählen und Passwort eingeben.' },
  'auth.checking':             { en: 'Checking…',              ru: 'Проверяю…',                de: 'Prüfe…' },
  'auth.wrong_password':       { en: 'Wrong password.',        ru: 'Пароль не подходит.',      de: 'Falsches Passwort.' },
  'auth.error_http':           { en: 'Error: HTTP {status}',   ru: 'Ошибка: HTTP {status}',    de: 'Fehler: HTTP {status}' },
  'auth.network_error':        { en: 'Network error: {err}',   ru: 'Сеть упала: {err}',        de: 'Netzwerkfehler: {err}' },
  'auth.record_profile_first': { en: 'First record a voice profile above.', ru: 'Сначала запиши голосовой профиль выше.', de: 'Erst ein Stimmprofil oben aufnehmen.' },
  'auth.select_and_new_password': { en: 'Select a profile and enter a new password.', ru: 'Выбери профиль и введи новый пароль.', de: 'Profil auswählen und neues Passwort eingeben.' },
  'auth.creating':             { en: 'Creating…',              ru: 'Создаю…',                  de: 'Erstelle…' },
  'auth.already_has_password': { en: 'This profile already has a password — log in and use «Change password».', ru: 'У профиля уже есть пароль — войди и смени из «Сменить пароль».', de: 'Dieses Profil hat bereits ein Passwort — anmelden und «Passwort ändern» verwenden.' },
  'auth.done_now_login':       { en: 'Done — you can now log in.', ru: 'Готово — теперь можешь войти.', de: 'Fertig — du kannst dich jetzt anmelden.' },

  // ── Pending actions panel ─────────────────────────────────────────
  'pending.loading':           { en: 'Loading…',               ru: 'Загружаю…',                de: 'Lädt…' },
  'pending.error':             { en: 'Error: {err}',           ru: 'Ошибка: {err}',            de: 'Fehler: {err}' },
  // {min} = minutes remaining until expiry
  'pending.expires_in_min':    { en: 'expires in {min} min',   ru: 'истекает через {min} мин', de: 'läuft in {min} Min. ab' },
  // {h} = hours remaining
  'pending.valid_for_h':       { en: 'valid for {h}h',         ru: 'действительно ещё {h} ч',  de: 'noch {h}h gültig' },
  // Recent section header
  'pending.recent_header':     { en: 'Recent:',                ru: 'Недавно:',                 de: 'Kürzlich:' },
  // Status labels for finalised pending actions
  'pending.status.executed':   { en: 'done',           ru: 'выполнено',  de: 'erledigt' },
  'pending.status.execution_failed': { en: 'failed',   ru: 'сбой',       de: 'fehlgeschlagen' },
  'pending.status.rejected':   { en: 'rejected',       ru: 'отклонено',  de: 'abgelehnt' },
  'pending.status.expired':    { en: 'expired',        ru: 'истекло',    de: 'abgelaufen' },

  // ── Memory panel ──────────────────────────────────────────────────
  'memory.loading':            { en: 'Loading…',               ru: 'Загружаю…',                de: 'Lädt…' },
  'memory.saving':             { en: 'Saving…',                ru: 'Сохраняю…',                de: 'Speichere…' },
  'memory.saved':              { en: 'Saved.',                  ru: 'Сохранено.',               de: 'Gespeichert.' },
  'memory.error':              { en: 'Error: {err}',           ru: 'Ошибка: {err}',            de: 'Fehler: {err}' },

  // ── Settings panel ─────────────────────────────────────────────────
  'settings.loading':          { en: 'Loading…',               ru: 'Загружаю…',                de: 'Lädt…' },
  'settings.saving':           { en: 'Saving…',                ru: 'Сохраняю…',                de: 'Speichere…' },
  'settings.saved':            { en: 'Saved.',                  ru: 'Сохранено.',               de: 'Gespeichert.' },
  'settings.error':            { en: 'Error: {err}',           ru: 'Ошибка: {err}',            de: 'Fehler: {err}' },
  'settings.json_broken':      { en: 'Broken JSON: {err}',     ru: 'JSON битый: {err}',        de: 'Ungültiges JSON: {err}' },
  // For the attach file loading error
  'attach.load_error':         { en: 'Failed to read file: {err}', ru: 'ошибка чтения файла: {err}', de: 'Datei konnte nicht gelesen werden: {err}' },
  // For skipping non-image files (dev log — kept for completeness)
  // For speaker load errors
  'speaker.profiles_load_error': { en: 'Error loading profiles: {err}', ru: 'Ошибка загрузки профилей: {err}', de: 'Fehler beim Laden der Profile: {err}' },
  'voice.clones_load_error':   { en: 'Error loading custom voices: {err}', ru: 'Ошибка загрузки кастомных голосов: {err}', de: 'Fehler beim Laden der eigenen Stimmen: {err}' },

  // ── PTT / attach panel text ──────────────────────────────────────
  'ptt.hint_pre':              { en: 'Hold the button (or', ru: 'Зажми и держи кнопку (или клавишу', de: 'Taste (oder' },
  'ptt.hint_post':             { en: '), speak, release — command skips the wake word.', ru: '), говори, отпусти — команда минует wake-слово.', de: ') halten, sprechen, loslassen — Befehl überspringt das Aktivierungswort.' },
  'attach.btn_title':          { en: 'Attach image',           ru: 'Прикрепить картинку',      de: 'Bild anhängen' },
  'attach.hint':               {
    en: 'Image will attach to the next voice turn — Claude will look at it and answer your question.',
    ru: 'Картинка прикрепится к следующей голосовой реплике — модель Claude посмотрит на неё и ответит на твой вопрос.',
    de: 'Bild wird an die nächste Sprachrunde angehängt — Claude schaut es an und beantwortet deine Frage.',
  },

  // ── Speaker panel (details summary / hints) ──────────────────────
  'speaker.panel_title':       { en: 'Voice profiles',         ru: 'Профили голоса',           de: 'Stimmprofile' },
  'speaker.name_placeholder':  { en: 'Your name',              ru: 'Твоё имя',                 de: 'Dein Name' },
  'speaker.record_btn':        { en: '🎤 Record (5 s)',        ru: '🎤 Записать (5 с)',         de: '🎤 Aufnehmen (5 s)' },
  'speaker.enroll_hint':       { en: 'Enter a name and press «Record» — speak for 5 seconds', ru: 'Введи имя и нажми «Записать» — говори 5 секунд', de: 'Namen eingeben und «Aufnehmen» drücken — 5 Sekunden sprechen' },

  // ── Custom voice panel (details summary / hints) ─────────────────
  'voice.panel_title':         { en: 'Output voices',          ru: 'Голоса для вывода',        de: 'Ausgabestimmen' },
  'voice.panel_hint':          {
    en: 'Record 6–12 seconds of any voice (yours, family, a character) — the assistant will reply with it. Voice per profile is chosen in «Voice profiles».',
    ru: 'Запиши 6–12 секунд любого голоса (свой, родных, персонажа) — ассистент будет отвечать им. Голос для каждого профиля выбирается в разделе «Профили голоса».',
    de: 'Nimm 6–12 Sekunden einer beliebigen Stimme auf (deine eigene, Familie, Charakter) — der Assistent antwortet damit. Stimme pro Profil wird in «Stimmprofile» gewählt.',
  },
  'voice.name_placeholder':    { en: 'Name (e.g. «Wife»)',     ru: 'Название (например, «Жена»)', de: 'Name (z. B. «Frau»)' },
  'voice.record_btn':          { en: '🎤 Record (6 s)',        ru: '🎤 Записать (6 с)',         de: '🎤 Aufnehmen (6 s)' },
  'voice.clone_hint':          { en: 'Enter a name and press «Record» — speak naturally for 6 seconds', ru: 'Введи название и нажми «Записать» — говори 6 секунд естественно', de: 'Namen eingeben und «Aufnehmen» drücken — 6 Sekunden natürlich sprechen' },

  // ── Pending panel (details summary) ─────────────────────────────
  'pending.panel_title':       { en: 'Deferred actions',       ru: 'Отложенные действия',      de: 'Aufgeschobene Aktionen' },

  // ── Memory panel (details summary) ─────────────────────────────
  'memory.panel_title':        { en: 'Memory',                 ru: 'Память',                   de: 'Speicher' },

  // ── Settings panel (details summary + labels) ────────────────────
  'settings.panel_title':      { en: 'Settings',               ru: 'Настройки',                de: 'Einstellungen' },
  'settings.panel_hint':       {
    en: 'Typed profile preferences: language, formality, voice, style. The <code>custom</code> field is a free-form object. <code>code_word_hash</code> is not editable here — use «Change password» in the header.',
    ru: 'Типизированные предпочтения профиля: язык, формальность, голос, стиль. Поле <code>custom</code> — свободный объект для чего угодно. Поле <code>code_word_hash</code> здесь не правится — для смены пароля используй кнопку «Сменить пароль» в шапке.',
    de: 'Typisierte Profileinstellungen: Sprache, Formalität, Stimme, Stil. Das <code>custom</code>-Feld ist ein freies Objekt. <code>code_word_hash</code> ist hier nicht bearbeitbar — nutze «Passwort ändern» im Header.',
  },
  'settings.label_language':   { en: 'Language',               ru: 'Язык',                     de: 'Sprache' },
  'settings.label_tone':       { en: 'Tone',                   ru: 'Тон',                      de: 'Ton' },
  'settings.label_voice':      { en: 'Voice (XTTS speaker name or <code>clone:&lt;id&gt;</code>)', ru: 'Голос (XTTS speaker name или <code>clone:&lt;id&gt;</code>)', de: 'Stimme (XTTS-Sprecher oder <code>clone:&lt;id&gt;</code>)' },
  'settings.label_style':      { en: 'Additional style for the assistant', ru: 'Дополнительный стиль для ассистента', de: 'Zusätzlicher Stil für den Assistenten' },
  'settings.style_placeholder': { en: '«reply briefly», «no jokes», «no profanity» …', ru: '«отвечай кратко», «без шуток», «не используй мат» …', de: '«antworte kurz», «keine Witze», «kein Schimpfwort» …' },
  'settings.raw_title':        { en: 'Full JSON (raw)',         ru: 'Полный JSON (raw)',         de: 'Vollständiges JSON (roh)' },
  'settings.raw_hint':         {
    en: 'Edit the structure freely — validated on save by the server (Pydantic). The <code>custom</code> section can serve as a scratch pad for arbitrary keys.',
    ru: 'Редактируй структуру свободно — при сохранении валидируется на сервере (Pydantic). Раздел <code>custom</code> можно использовать как блокнот для произвольных ключей.',
    de: 'Struktur frei bearbeiten — wird beim Speichern vom Server validiert (Pydantic). Der <code>custom</code>-Abschnitt kann als Notizbuch für beliebige Schlüssel dienen.',
  },

  // ── Stats panel (details summary + labels) ─────────────────────
  'stats.panel_title':         { en: 'Statistics',             ru: 'Статистика',               de: 'Statistik' },
  'stats.label_daily':         { en: 'Daily spend (prompt vs completion)', ru: 'Расход по дням (prompt vs completion)', de: 'Tagesausgaben (Prompt vs. Completion)' },
  'stats.label_per_tool':      { en: 'Spend by tool',          ru: 'Расход по инструментам',   de: 'Ausgaben nach Tool' },
  'stats.label_per_client':    { en: 'Spend by client',        ru: 'Расход по клиентам',       de: 'Ausgaben nach Client' },

  // ── Log panel header ────────────────────────────────────────────
  'log.panel_header':          { en: 'Log',                    ru: 'Лог',                      de: 'Log' },

  // ── Auth panel headers and hints ────────────────────────────────
  'auth.anonymous_header':     { en: 'Sign in',                ru: 'Вход',                     de: 'Anmelden' },
  'auth.anonymous_hint':       {
    en: 'Log in with your profile to edit memory, settings, and confirm deferred actions.',
    ru: 'Войди под своим профилем, чтобы редактировать память, настройки и подтверждать отложенные действия.',
    de: 'Mit Profil anmelden, um Speicher, Einstellungen zu bearbeiten und aufgeschobene Aktionen zu bestätigen.',
  },

  // ── Connected devices (Wave 2 Phase 4 agents panel) ──────────────
  // Read-only panel listing every desktop-agent the orchestrator can
  // see, with capability badges and an online/offline indicator.
  'agents.header': {
    en: 'Connected devices',
    ru: 'Подключенные устройства',
    de: 'Verbundene Geräte',
  },
  'agents.empty':  {
    en: 'No agents registered.',
    ru: 'Нет подключенных устройств.',
    de: 'Keine Geräte registriert.',
  },
  'agents.online':  { en: 'online',  ru: 'на связи',     de: 'online' },
  'agents.offline': { en: 'offline', ru: 'не на связи',  de: 'offline' },
  'agents.default_label': {
    en: 'default', ru: 'по умолчанию', de: 'Standard',
  },
  // Capability badges — short labels next to each device row.  Kept
  // terse because they sit inline with an emoji prefix.
  'agents.cap_screenshot':   { en: 'screenshot',  ru: 'скриншот',        de: 'Screenshot' },
  'agents.cap_applescript':  { en: 'AppleScript', ru: 'AppleScript',     de: 'AppleScript' },
  'agents.cap_pyautogui':    { en: 'mouse/keys',  ru: 'мышь/клавиатура', de: 'Maus/Tasten' },
  'agents.cap_hotkey':       { en: 'hotkey',      ru: 'клавиши',         de: 'Tasten' },
  'agents.cap_default_apps': { en: 'default apps', ru: 'приложения',     de: 'Standard-Apps' },
  'agents.cap_cursor':       { en: 'cursor',      ru: 'курсор',          de: 'Cursor' },
  // Platform labels — used in parens after the agent_id.
  'agents.platform_macos':   { en: 'macOS',   ru: 'macOS',   de: 'macOS' },
  'agents.platform_windows': { en: 'Windows', ru: 'Windows', de: 'Windows' },
  'agents.platform_linux':   { en: 'Linux',   ru: 'Linux',   de: 'Linux' },
  // `{n}` = seconds since the last tool call routed to this agent.
  // Rendered as a small chip on the matching row in the agents panel
  // so the user can see which device just acted on their voice turn.
  'agents.active_recent': {
    en: 'active {n}s ago',
    ru: 'активен {n}с назад',
    de: 'aktiv vor {n}s',
  },

  // ── Item store (Phase 2) ──────────────────────────────────────────────
  'items.panel_title':       { en: 'Item Store',      ru: 'Хранилище',          de: 'Ablage' },
  'items.all':               { en: 'All items',       ru: 'Все элементы',       de: 'Alle Einträge' },
  'items.search_placeholder':{ en: 'Search items…',  ru: 'Поиск…',             de: 'Suchen…' },
  'items.kind_text':         { en: 'Note',            ru: 'Заметка',            de: 'Notiz' },
  'items.kind_link':         { en: 'Link',            ru: 'Ссылка',             de: 'Link' },
  'items.kind_video':        { en: 'Video',           ru: 'Видео',              de: 'Video' },
  'items.add_placeholder':   { en: 'Text or URL…',   ru: 'Текст или ссылка…', de: 'Text oder URL…' },
  'items.add_btn':           { en: 'Add',             ru: 'Добавить',           de: 'Hinzufügen' },
  'items.loading':           { en: 'Loading…',        ru: 'Загружаю…',          de: 'Lädt…' },
  'items.empty':             { en: 'No items here.',  ru: 'Элементов нет.',     de: 'Keine Einträge.' },
  'items.trash_empty':       { en: 'Trash is empty.', ru: 'Корзина пуста.',    de: 'Papierkorb leer.' },
  'items.show_trash':        { en: '🗑 Trash',         ru: '🗑 Корзина',         de: '🗑 Papierkorb' },
  'items.hide_trash':        { en: '← Back',          ru: '← Назад',           de: '← Zurück' },
  'items.restore_btn':       { en: 'Restore',         ru: 'Восстановить',      de: 'Wiederherstellen' },
  'items.delete_btn':        { en: 'Delete',          ru: 'Удалить',            de: 'Löschen' },
  'items.new_folder':        { en: 'Folder',          ru: 'Папка',              de: 'Ordner' },
  'items.new_checklist':     { en: 'List',            ru: 'Список',             de: 'Liste' },
  'items.cat_rename_title':  { en: 'Rename',          ru: 'Переименовать',      de: 'Umbenennen' },
  'items.cat_delete_title':  { en: 'Delete category', ru: 'Удалить категорию', de: 'Kategorie löschen' },
  'items.cat_name_prompt':   { en: 'Category name:',  ru: 'Имя категории:',    de: 'Kategoriename:' },
  'items.rename_prompt':     { en: 'New name:',        ru: 'Новое имя:',        de: 'Neuer Name:' },
  'items.cat_delete_confirm': {
    en: 'Delete "{name}"? Items inside are preserved but hidden.',
    ru: 'Удалить "{name}"? Элементы внутри сохранятся, но станут недоступны.',
    de: '"{name}" löschen? Enthaltene Einträge bleiben erhalten, sind aber verborgen.',
  },
  'items.auto_sort_btn':    { en: '✨ Sort with AI', ru: '✨ AI-сортировка',  de: '✨ KI-Sortierung' },
  'items.auto_sort_title':  { en: 'Suggested moves',  ru: 'Предложенные перемещения', de: 'Vorgeschlagene Verschiebungen' },
  'items.auto_sort_none':   { en: 'No suggestions — items already look well-categorised.', ru: 'Предложений нет — элементы уже хорошо распределены.', de: 'Keine Vorschläge — Einträge sind bereits gut kategorisiert.' },
  'items.auto_sort_apply':  { en: 'Apply',            ru: 'Применить',          de: 'Anwenden' },
  'items.auto_sort_cancel': { en: 'Cancel',           ru: 'Отмена',             de: 'Abbrechen' },
  'items.error':            { en: 'Error',            ru: 'Ошибка',             de: 'Fehler' },

  // ── Live stream panel ───────────────────────────────────────────────
  'stream.stop':        { en: 'Stop',         ru: 'Стоп',          de: 'Stopp' },
  'stream.label_camera': { en: '📷 Live — Camera', ru: '📷 Эфир — Камера', de: '📷 Live — Kamera' },
  'stream.label_tab':    { en: '🖥 Live — Tab',   ru: '🖥 Эфир — Вкладка', de: '🖥 Live — Tab' },
};


function _readLang() {
  const ls = localStorage.getItem('va_lang');
  if (ls && SUPPORTED_LANGS.includes(ls)) return ls;
  const nav = (navigator.language || 'en').slice(0, 2).toLowerCase();
  if (SUPPORTED_LANGS.includes(nav)) return nav;
  return DEFAULT_LANG;
}

let _currentLang = _readLang();

export function currentLang() { return _currentLang; }

export function setLang(lang) {
  if (!SUPPORTED_LANGS.includes(lang)) return;
  _currentLang = lang;
  localStorage.setItem('va_lang', lang);
  applyI18n();
}

/** Look up a translated string.  Falls back to English, then to the key itself.
 *
 * Second arg can be either ``lang`` (string) for back-compat with
 * existing callers, or a ``{...}`` object of named placeholders (e.g.
 * ``{n: 3}``) which get spliced in using Python-style ``{name}``
 * substitution.  When you need both — uncommon — pass an object with
 * a ``_lang`` field.
 */
export function t(key, langOrFmt) {
  let L = _currentLang;
  let fmt = null;
  if (typeof langOrFmt === 'string') {
    L = langOrFmt;
  } else if (langOrFmt && typeof langOrFmt === 'object') {
    fmt = langOrFmt;
    if (typeof fmt._lang === 'string') L = fmt._lang;
  }
  const entry = CATALOG[key];
  if (!entry) return key;
  let text = entry[L] || entry[DEFAULT_LANG] || key;
  if (fmt) {
    text = text.replace(/\{(\w+)\}/g, (_, k) => (k in fmt ? String(fmt[k]) : `{${k}}`));
  }
  return text;
}

/**
 * Walk every element with `data-i18n="<key>"` and replace its textContent
 * with `t(key)`.  Run once at boot and again any time the language
 * changes.  Cheap — a few dozen elements at most.
 */
export function applyI18n() {
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const key = el.getAttribute('data-i18n');
    if (key) el.textContent = t(key);
  });
  // Also handle `data-i18n-placeholder` for inputs and `data-i18n-title`
  // for tooltip-bearing buttons — both common cases.
  document.querySelectorAll('[data-i18n-placeholder]').forEach((el) => {
    const key = el.getAttribute('data-i18n-placeholder');
    if (key) el.placeholder = t(key);
  });
  document.querySelectorAll('[data-i18n-title]').forEach((el) => {
    const key = el.getAttribute('data-i18n-title');
    if (key) el.title = t(key);
  });
}

/** Pull user's settings.language from /api/me's settings and align with us.
 *  Called after a successful login.  "auto" leaves the current value alone
 *  (so navigator-detected or manually-chosen wins). */
export function syncFromSettings(settingsLang) {
  if (!settingsLang || settingsLang === 'auto') return;
  if (SUPPORTED_LANGS.includes(settingsLang) && settingsLang !== _currentLang) {
    setLang(settingsLang);
  }
}
