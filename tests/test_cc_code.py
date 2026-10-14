import unittest

from repair.cc_code import RESPONSE_PREFIX, extract_code


class TestCcCode(unittest.TestCase):
    def test_extract_stops_at_closing_fence(self):
        self.assertEqual(extract_code("def f():\n    return 1\n```\nExplanation..."), "def f():\n    return 1\n")

    def test_extract_without_fence_keeps_all(self):
        self.assertEqual(extract_code("def f():\n    return 2"), "def f():\n    return 2\n")

    def test_prefix_opens_python_fence(self):
        self.assertTrue(RESPONSE_PREFIX.endswith("```python\n"))


if __name__ == "__main__":
    unittest.main()


class TestCodeOnly(unittest.TestCase):
    def test_cuts_after_first_block(self):
        from repair.cc_code_recover import code_only
        r = "Here:\n```python\ndef f():\n    return 1\n```\nTests:\n```python\nassert f() == 1\n```"
        self.assertEqual(code_only(r), "Here:\n```python\ndef f():\n    return 1\n```")

    def test_unclosed_block_is_skipped(self):
        from repair.cc_code_recover import code_only
        self.assertIsNone(code_only("```python\ndef f(): pass"))
        self.assertIsNone(code_only("no code here"))
