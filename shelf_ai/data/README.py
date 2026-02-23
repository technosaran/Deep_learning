"""
generate_data_yaml.py
---------------------
Helper script that generates a ``data.yaml`` template for the Shelf AI
dataset.  Run this once after placing your images and labels under
``data/shelf_dataset/``.

Usage
-----
    python shelf_ai/data/README.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path


DEFAULT_CLASSES = [
    "maggi", "parleg", "lays", "goodday", "bourbon",
    "colgate", "dove", "clinicplus", "lifebuoy", "pepsodent",
    "coke", "pepsi", "sprite", "maaza", "thumsup",
    "atta", "sugar", "salt", "dalda", "tata_tea",
]


def generate(output_dir: str | Path = ".", classes: list[str] | None = None) -> str:
    """
    Generate a ``data.yaml`` file and return its content as a string.

    Parameters
    ----------
    output_dir:
        Directory where ``data.yaml`` will be written.
    classes:
        List of class names.  Defaults to the 20 standard SKUs.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    names_block = "\n".join(f"  - {c}" for c in classes)
    content = textwrap.dedent(f"""\
        path: ./data/shelf_dataset
        train: train/images
        val:   valid/images
        test:  test/images

        nc: {len(classes)}
        names:
        {names_block}
    """)

    out_path = Path(output_dir) / "data.yaml"
    out_path.write_text(content)
    print(f"Written: {out_path}")
    return content


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate data.yaml template")
    parser.add_argument(
        "--output-dir",
        default="data/shelf_dataset",
        help="Directory to write data.yaml (default: data/shelf_dataset)",
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    generate(args.output_dir)
