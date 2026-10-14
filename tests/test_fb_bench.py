"""Pilot 5 harness: the MultiPL-E runner must pass a correct body and fail a wrong one; MMLU letter picking."""
import json
import shutil
import unittest
from pathlib import Path

from repair.fb_bench import DATA, mmlu_choice, run_program

BODIES = {  # MultiPL-E problem 0 (has_close_elements), completion = function body after the prompt
    "js": ("  for (let i = 0; i < numbers.length; i++)\n    for (let j = i + 1; j < numbers.length; j++)\n"
           "      if (Math.abs(numbers[i] - numbers[j]) < threshold) return true;\n  return false;\n}\n", "  return false;\n}\n"),
    "cpp": ("    for (size_t i = 0; i < numbers.size(); i++)\n        for (size_t j = i + 1; j < numbers.size(); j++)\n"
            "            if (std::abs(numbers[i] - numbers[j]) < threshold) return true;\n    return false;\n", "    return false;\n"),
}


@unittest.skipUnless((DATA / "multipl_e_js.json").exists(), "run `python -m repair.fb_bench prep` first")
class TestMultiplE(unittest.TestCase):
    def check(self, lang):
        p = json.loads((DATA / f"multipl_e_{lang}.json").read_text())[0]
        good, bad = BODIES[lang]
        self.assertTrue(run_program(lang, p["prompt"] + good + "\n" + p["tests"]))
        self.assertFalse(run_program(lang, p["prompt"] + bad + "\n" + p["tests"]))

    @unittest.skipUnless(shutil.which("node"), "node missing")
    def test_js(self):
        self.check("js")

    @unittest.skipUnless(shutil.which("g++"), "g++ missing")
    def test_cpp(self):
        self.check("cpp")


class TestMmluChoice(unittest.TestCase):
    def test_picks_most_likely_letter_with_or_without_space(self):
        self.assertEqual(mmlu_choice({" B": -0.1, " A": -2.0, "C": -0.05, "hello": 0.0}), 2)

    def test_none_when_no_letter(self):
        self.assertIsNone(mmlu_choice({"hello": -0.1}))


if __name__ == "__main__":
    unittest.main()
