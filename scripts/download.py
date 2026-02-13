#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path


def download_modelscope(model_id: str, local_dir: Path):
   from modelscope import snapshot_download
   snapshot_download(
       model_id=model_id,
       local_dir=str(local_dir)
   )


def download_huggingface(model_id: str, local_dir: Path):
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id=model_id,
       local_dir=str(local_dir)
   )


def main():
   parser = argparse.ArgumentParser(description='Download model')
   parser.add_argument('-m', '--model_id', required=True, help='Model ID')
   parser.add_argument('--source', choices=['modelscope', 'huggingface'], 
                      default='modelscope', help='Download source')
   args = parser.parse_args()

   model_name = args.model_id.split('/')[-1]
   script_dir = Path(__file__).parent.absolute()
   root_dir = script_dir.parent
   local_dir = root_dir / 'models' / model_name
   local_dir.mkdir(parents=True, exist_ok=True)

   if args.source == 'modelscope':
       download_modelscope(args.model_id, local_dir)
   else:
       download_huggingface(args.model_id, local_dir)

   print(f"Downloaded to: {local_dir}")


if __name__ == "__main__":
   main()