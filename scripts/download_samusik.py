from __future__ import annotations

import argparse
import os
from pathlib import Path
import urllib.request


URL = "https://experimenthub.bioconductor.org/fetch/2246"
EXPECTED_BYTES = 192_164_364


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Samusik_all from ExperimentHub")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/samusik/Samusik_all_SE.rda"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_file() and output.stat().st_size == EXPECTED_BYTES:
        print(f"already present: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(URL, headers={"User-Agent": "StrataSCAN benchmark"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)
    if temporary.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(
            f"unexpected download size: {temporary.stat().st_size} != {EXPECTED_BYTES}"
        )
    os.replace(temporary, output)
    print(f"downloaded: {output}")


if __name__ == "__main__":
    main()
