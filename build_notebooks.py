#!/usr/bin/env python

import sys
import os
import subprocess
import pathlib


def main():
    root = pathlib.Path()
    for notebook in root.rglob('*.ipynb'):
        parent_dir = notebook.parent
        command = f'jupyter nbconvert "{notebook.name}" --to markdown'
        if not (parent_dir / f"{notebook.stem}.md").exists():
            print(f"~$ {command} in {parent_dir}")
            subprocess.run(command, shell=True, cwd=parent_dir)


if __name__ == '__main__':
    sys.exit(main())