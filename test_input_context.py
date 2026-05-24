import unittest

from input_context import describe_datetime


class DescribeDatetimeTests(unittest.TestCase):
    def test_describe_datetime_exposes_hours_and_minutes(self):
        result = describe_datetime(
            "2024-11-15T10:12:00",
            current_time="2024-11-15T10:12:00",
        )

        self.assertEqual(result["whole_hours"], 10)
        self.assertEqual(result["whole_minutes"], 12)
        self.assertEqual(result["hours_until_workday_start"], 0)

    def test_describe_datetime_adds_hours_until_workday_start_when_outside_hours(self):
        result = describe_datetime(
            "2024-11-15T01:30:00",
            current_time="2024-11-15T01:30:00",
        )

        self.assertEqual(result["hours_until_workday_start"], 6)
        self.assertEqual(result["is_standard_working_hours"], False)


if __name__ == "__main__":
    unittest.main()
