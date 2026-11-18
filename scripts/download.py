"""Download the prepared datasets from Hugging Face into one folder per dataset, the layout the db container mounts
read-only as /data:

  <dest>/<dataset>/{train.csv, test.csv, ground_truth.csv, manifest.json}

Files that already exist are not downloaded again. The train.csv files are 1 to 5 GB each. On the test server the
files are already in its filesystem, so this step is only needed elsewhere.
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from lancedb_bench.config import DATASET_FILES, HF_REPO, add_dataset_arg, select_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(parser)
    parser.add_argument("--dest", default="./data", help="creates <dest>/<dataset>/ (default ./data)")
    args = parser.parse_args()

    for ds in select_datasets(args.dataset):
        for name in DATASET_FILES:
            target = Path(args.dest) / ds.name / name
            if target.exists():
                print(f"{target}: already there")
                continue
            hf_hub_download(repo_id=HF_REPO, filename=f"{ds.name}/{name}", repo_type="dataset", local_dir=args.dest)
            print(f"{target}: downloaded")


if __name__ == "__main__":
    main()
