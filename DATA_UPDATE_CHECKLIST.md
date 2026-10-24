# ATP data update checks

Run the season CSV refresh, results sync, snapshot rebuild, and
`scripts/audit_live_snapshot.py` before publishing either the API or web data.

1. Compare the day's completed main-draw matches and upcoming fixtures with
   current ATP/tournament draws and order of play. Check pairings, round, start
   time, retirement/default, and cancellation status.
2. Stop on every `AUDIT REVIEW` or `AUDIT FAIL`. The audit checks qualifying
   leaks, unknown rounds, semifinal coverage after four QFs, and final coverage
   after both SFs. Do not publish before resolving these against sources.
3. Ensure Previous contains only predictions captured before match start.
   Never fabricate or backfill a prediction after a result is known.
4. If both feeds omit a real match, use a verified fixture override only with
   source URLs, confirmed entrants from preceding-round results, and a real
   start time. Report any unverified source gap instead of guessing.
