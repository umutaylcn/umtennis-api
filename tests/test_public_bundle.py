import unittest

from scripts.validate_public_bundle import validate


def bundle(upcoming=(1,), previous=(2,)):
    return {
        "upcoming": {
            "count": len(upcoming),
            "matches": [{"match_id": match_id} for match_id in upcoming],
            "predictions": {str(match_id): {} for match_id in upcoming},
        },
        "previous": {
            "count": len(previous),
            "matches": [{"match_id": match_id} for match_id in previous],
        },
    }


class PublicBundleTests(unittest.TestCase):
    def test_valid_bundle(self):
        validate(bundle(), bundle(previous=()))

    def test_completed_match_cannot_remain_upcoming(self):
        with self.assertRaisesRegex(ValueError, "Completed match"):
            validate(bundle(upcoming=(1,), previous=(1,)))

    def test_previous_results_cannot_disappear(self):
        with self.assertRaisesRegex(ValueError, "remove existing"):
            validate(bundle(previous=(2,)), bundle(previous=(2, 3)))

    def test_prediction_must_match_fixture(self):
        item = bundle()
        item["upcoming"]["predictions"] = {}
        with self.assertRaisesRegex(ValueError, "predictions"):
            validate(item)


if __name__ == "__main__":
    unittest.main()
