import unittest

from intent_server import _wait_for_service_calls


class ServiceCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_successful_call(self):
        completed = []

        async def succeed():
            completed.append(True)

        await _wait_for_service_calls([succeed()])

        self.assertEqual([True], completed)

    async def test_all_failed_calls_raise(self):
        async def fail(message):
            raise RuntimeError(message)

        with self.assertRaisesRegex(RuntimeError, "first"):
            await _wait_for_service_calls([fail("first"), fail("second")])

    async def test_partial_failure_still_counts_as_handled(self):
        async def succeed():
            return None

        async def fail():
            raise RuntimeError("partial")

        await _wait_for_service_calls([fail(), succeed()])


if __name__ == "__main__":
    unittest.main()
