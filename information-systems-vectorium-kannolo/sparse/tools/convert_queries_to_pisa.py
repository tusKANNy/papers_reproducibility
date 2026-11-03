
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert sparse JSONL queries to PISA format")
    parser.add_argument("input", help="Input JSONL file with encoded queries")
    parser.add_argument("output", help="Output .pisa query file")
    parser.add_argument("--scale", type=float, default=1000.0,
                        help="Multiply scores by this factor before rounding (default: 1000)")
    args = parser.parse_args()

    print(f"Loading queries from {args.input}...", file=sys.stderr)
    n = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            qid = obj["id"]
            tokens = []
            for token, score in sorted(obj["vector"].items(), key=lambda x: -x[1]):
                count = round(score * args.scale)
                if count > 0:
                    tokens.extend([token] * count)
            fout.write(f"{qid}:{' '.join(tokens)}\n")
            n += 1
            if n % 1000 == 0:
                print(f"  Processed {n} queries", file=sys.stderr)

    print(f"Queries written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
