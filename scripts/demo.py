"""Answer a single question end-to-end and print the cited sources.

  python scripts/demo.py "Who was the maternal grandfather of Nancy Ainsworth?"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genealogy_rag import GenealogyRAG  # noqa: E402


def main() -> None:
    query = " ".join(sys.argv[1:]) or \
        "Who was the maternal grandfather of Nancy Ainsworth?"
    rag = GenealogyRAG()
    ans = rag.answer(query, k=5)
    print(ans.render())
    top = ans.context[0] if ans.context else None
    if top:
        print(f"\nTop source text:\n  {top.text}")


if __name__ == "__main__":
    main()
