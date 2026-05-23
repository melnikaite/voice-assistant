# Adding a new locale

**One JSON file. No Python edits.**

`orchestrator/app/i18n.py` loads every `orchestrator/app/locales/*.json`
at startup. Each file holds the language's complete locale data:
messages, currencies, weather phrases, intent regexes, ASR hint.
Number-to-words and date/time formatting come from `num2words` and
`Babel` — both already know 40+ languages, including yours.

## Recipe

1. Copy `orchestrator/app/locales/en.json` to
   `orchestrator/app/locales/<your-code>.json` (e.g. `fr.json`).
   The code MUST be a valid 2-letter ISO 639-1 — `num2words` and
   `Babel` key off it.
2. Edit `_meta`:
   ```json
   "_meta": {
     "code": "fr",
     "name": "Français",
     "babel_locale": "fr_FR",
     "num2words_lang": "fr"
   }
   ```
3. Translate every value under `messages`, `currencies.names`,
   `weather_codes`, `intents.*`, and the `asr.whisper_hint` message.
4. Restart the orchestrator. The locale picker in the frontend will
   pick up the new file automatically.

That's it.

## File schema

```json
{
  "_meta": { "code": "...", "babel_locale": "...", "num2words_lang": "..." },
  "messages":      { "<key.path>": "<template with {placeholders}>" },
  "currencies": {
    "names":    { "<ISO>": "<spoken plural noun>" },
    "aliases":  { "<spoken phrase>": "<ISO>" }
  },
  "weather_codes": { "<WMO code>": "<phrase>" },
  "intents":       { "<intent_name>": [ "<regex>", ... ] }
}
```

### `messages`

Key → template with Python `str.format` placeholders. Use named
placeholders (`{when}`, never `{0}`) so translators can reorder.

Any key missing in your locale falls back to English — a half-
translated locale is still usable; users see EN for whatever isn't
done yet.

### `currencies`

- `names` — ISO code → spoken plural noun ("500 EUR" → "500 euros").
- `aliases` — spoken phrase → ISO. Lowercase keys; the matcher
  lowercases input before lookup.

### `weather_codes`

WMO code → phrase. See [Open-Meteo's table](https://open-meteo.com/en/docs#weathervariables);
copy from `en.json` and translate.

### `intents`

Compiled regex patterns for zero-LLM phrase matching:

| Intent              | Purpose                                                   | Locale scope      |
|---------------------|-----------------------------------------------------------|-------------------|
| `replay_resume`     | "Continue from where you stopped" — TTS resumes.          | active locale     |
| `replay_full`       | "Repeat that" — re-speaks the last reply.                 | active locale     |
| `new_topic`         | "New topic" — clears conversation history.                | active locale     |
| `voicemail`         | "Tell Anna that ..." — captures `(?P<to>)`, `(?P<body>)`. | ALL locales       |
| `passphrase_prefix` | "Password: ..." — captures `(?P<phrase>)`.                | ALL locales       |
| `destructive_text`  | Click-target safety filter (Delete / Send / Empty / …).   | ALL locales       |

"ALL locales" means the orchestrator matches across every loaded
locale's patterns — so a Russian-locale user saying "delete this" in
English still trips the destructive-text gate.

Regex syntax: Python `re` flavour, compiled with `IGNORECASE`. Double-
escape backslashes (`\\b` for word boundary).

## What you get for free

| Capability        | Source                                                        |
|-------------------|---------------------------------------------------------------|
| Number-to-words   | `num2words(n, lang=<your code>)` — 40+ languages.             |
| Duration phrases  | Babel `format_timedelta(locale=<your locale>)`.               |
| Relative time     | Babel `format_datetime` with `add_direction`.                 |
| Wake-word         | Language-agnostic (audio classifier).                         |
| ASR (Whisper)     | Auto-detects language; `asr.whisper_hint` biases proper nouns.|
| TTS               | XTTS-v2 supports many languages — see `xtts-server`.          |
| Speaker-ID        | Language-agnostic (d-vectors).                                |

## Testing

1. Set the speaker's `language` to your code via the settings tool
   ("set my language to fr") or the frontend dropdown.
2. Exercise tools: calculator ("twelve plus four"), weather, set a
   timer, remember something, voicemail.
3. Any English you hear back means a `messages.<key>` is missing — fill in.
