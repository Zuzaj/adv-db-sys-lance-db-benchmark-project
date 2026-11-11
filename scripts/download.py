"""Download the prepared datasets from Hugging Face into the layout the containers mount:

  <dest>/db/<dataset>/train.csv                                      (db container)
  <dest>/client/<dataset>/{test.csv, ground_truth.csv, manifest.json}   (client container)

Files that are already complete are skipped. The train.csv files are 1 to 5 GB each.
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from lancedb_bench.config import HF_REPO, ROLE_FILES, add_dataset_arg, select_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(parser)
    parser.add_argument("--dest", default="./data", help="creates <dest>/db and <dest>/client (default ./data)")
    args = parser.parse_args()

    for ds in select_datasets(args.dataset):
        for role, files in ROLE_FILES.items():
            for name in files:
                path = hf_hub_download(repo_id=HF_REPO, filename=f"{ds.name}/{name}", repo_type="dataset", local_dir=Path(args.dest) / role)
                print(path)


if __name__ == "__main__":
    main()
