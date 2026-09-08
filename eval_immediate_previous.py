import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from graphService import analyzeImmediateRelationship


def main():
    casesPath = Path(__file__).with_name("eval_cases.json")
    cases = json.loads(casesPath.read_text())

    graded = [case for case in cases if not case.get("knownGap")]
    knownGaps = [case for case in cases if case.get("knownGap")]

    correct = 0
    for case in graded:
        result = analyzeImmediateRelationship(
            case["previousUserText"],
            case["previousAiText"],
            case["userText"],
        )
        predicted = result["selectedLabel"]
        matches = predicted == case["expectedLabel"]
        correct += int(matches)
        status = "PASS" if matches else "FAIL"
        print(
            f"{status:4}  {case['name']}: expected={case['expectedLabel']} "
            f"predicted={predicted} confidence={result['selectedConfidence']}"
        )

    print(f"\nAccuracy: {correct}/{len(graded)} = {correct / len(graded):.1%}")

    # Cases the bi-encoder can't currently resolve (documented, not graded).
    # If one starts passing, tighten it into a graded case and drop the flag.
    if knownGaps:
        print("\nKnown gaps (not graded):")
        for case in knownGaps:
            result = analyzeImmediateRelationship(
                case["previousUserText"],
                case["previousAiText"],
                case["userText"],
            )
            predicted = result["selectedLabel"]
            nowPasses = " -- NOW PASSES, promote it" if predicted == case["expectedLabel"] else ""
            print(
                f"      {case['name']}: expected={case['expectedLabel']} "
                f"predicted={predicted}{nowPasses}"
            )
            print(f"        {case['note']}")


if __name__ == "__main__":
    main()
