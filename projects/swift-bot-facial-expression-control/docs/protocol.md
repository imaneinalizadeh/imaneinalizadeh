# Client-Server Wire Protocol

Plain TCP, newline-delimited text.

## Format
```
COMMAND|gaze_tag\n
```
- `COMMAND` — one of `ADVANCE`, `RETREAT`, `HOLD`.
- `gaze_tag` — one of `left`, `centre`, `right`, `unknown`. Currently
  logged only by the server — reserved for a future steering extension.

## Example session
```
ADVANCE|centre
RETREAT|left
HOLD|centre
```

## Design decisions
- **Client de-duplicates before sending**: only sends a line when the
  command actually changes, not once per video frame.
- **Server is defensive regardless**: `CommandHandler` also checks the
  last command before dispatching, so a future client that doesn't
  de-duplicate won't spam the motors.
- **Unknown commands are logged and dropped, not fatal.**
