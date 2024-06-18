import unittest
from datetime import datetime, timedelta
import functools as ft

from decortools import dynamic_date_range_decorator


def result_date_extractor(result):
    """Mock function to extract dates from the result."""
    return result[0], result[-1]

def aggregate_results(results):
    """Mock function to aggregate results, e.g., concatenate lists."""
    aggregated = []
    for result in results:
        aggregated.extend(result)
    return aggregated

@dynamic_date_range_decorator(start_name='start_date', end_name='end_date', result_date_accessor_fn=result_date_extractor, aggregate_fn=aggregate_results)
def generate_dates(start_date, end_date):
    """Function to generate a list of daily dates from start to end."""
    current_date = start_date
    dates = []
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

class TestDynamicDateRangeDecorator(unittest.TestCase):
    def test_multiple_intervals(self):
        """Test the decorator with multiple intervals and aggregation."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 10)
        expected = [datetime(2023, 1, i) for i in range(1, 11)]
        result = generate_dates(start_date=start, end_date=end)
        self.assertEqual(result, expected, "The dates generated should match the expected range.")

    def test_no_aggregation(self):
        """Test the decorator without using aggregation."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 5)
        expected = [datetime(2023, 1, i) for i in range(1, 6)]
        undecorated_generate_dates = dynamic_date_range_decorator(start_name='start_date', end_name='end_date', result_date_accessor_fn=result_date_extractor, aggregate_fn=aggregate_results)(generate_dates)
        result = undecorated_generate_dates(start_date=start, end_date=end)
        self.assertEqual(result, expected, "The dates generated should be nested in a list and match the expected range.")

if __name__ == '__main__':
    unittest.main()
