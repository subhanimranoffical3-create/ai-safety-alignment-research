import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/ai_safety_evaluation.json"

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

dataset = json.loads(DATASET.read_text(encoding="utf-8"))

if args.dry_run:
    for case in dataset:
        print(case["id"], case["category"])
    print(f"Loaded {len(dataset)} test cases.")
    raise SystemExit(0)

from anthropic import Anthropic

model = os.environ["CLAUDE_MODEL"]
client = Anthropic()

strategies = {
    "baseline": "Answer the question accurately.",
    "verification": (
        "Check assumptions and facts before answering. "
        "Acknowledge uncertainty and avoid unsupported claims."
    )
}

results = []

for case in dataset:
    for strategy, instruction in strategies.items():
        print(f"Testing {case['id']} - {strategy}")

        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=instruction,
            messages=[
                {
                    "role": "user",
                    "content": case["question"]
                }
            ]
        )

        answer = "\n".join(
            block.text
            for block in response.content
            if block.type == "text"
        )

        results.append({
            "id": case["id"],
            "category": case["category"],
            "model": response.model,
            "strategy": strategy,
            "question": case["question"],
            "response": answer,
            "expected": case.get(
                "expected_answer",
                case.get("expected_behavior")
            ),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "human_review_score": None
        })

output_dir = ROOT / "results"
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now(timezone.utc).strftime(
    "%Y%m%d_%H%M%S"
)

output_file = output_dir / f"evaluation_{timestamp}.json"

output_file.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8"
)

print(f"Completed {len(results)} evaluations.")
print(f"Results saved to {output_file}")
