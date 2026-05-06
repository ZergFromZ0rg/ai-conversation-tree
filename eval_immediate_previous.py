import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from graphService import analyzeImmediateRelationship


def main():
    casesPath = Path(__file__).with_name("eval_cases.json")
    cases = json.loads(casesPath.read_text())

    correct = 0
    for case in cases:
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

    print(f"\nAccuracy: {correct}/{len(cases)} = {correct / len(cases):.1%}")


if __name__ == "__main__":
    main()
