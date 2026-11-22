from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from types import SimpleNamespace
from tennis_ai.results_backfill import match_identity_key


def test_turin_playoff_csv_and_feed_share_identity():
    fields = dict(played_at_utc="2026-11-21T13:40:00Z", winner_name="Carlos Alcaraz", loser_name="Ben Shelton", round="SF")
    assert match_identity_key(SimpleNamespace(tourney_name="Finals - Turin", **fields)) == match_identity_key(SimpleNamespace(tourney_name="Finals - Turin ATP - Play Offs (indoor)", **fields))

