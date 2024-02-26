#!/usr/bin/env python

import sys
import re
import pathlib
import subprocess


def main():
    root = pathlib.Path()
    for path in root.glob('*'):
        if path.is_dir():
            build_latex(path)
            build_index(path)
    remove_buildfiles(root)

def build_latex(path):
    print(f"Building notes for {path}\n")
    subprocess.run("make", shell=True, cwd=path)

def build_index(path):
    dir_name = split_camel_case(path.stem)
    template = f"""
---
layout: post
title: {dir_name} Notes
nav-menu: false
show_tile: false
---

"""
    for note_file in path.glob('*.pdf'):
        filename = note_file.stem
        note_name = split_camel_case(filename)
        template += f"""
# {note_name}

<iframe src="/college/{path.name}/{note_file.name}"
        style="width: 100%; height: 25em;">
</iframe>
"""
    (path / 'index.md').write_text(template)

def remove_buildfiles(path):
	subprocess.run(
        (
            'find . -type f'
            r' | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"'
            ' | xargs rm'
        ),
        shell=True,
        cwd=path,
    )



def split_camel_case(word):
    return re.sub('([A-Z0-9])', r' \1', word).strip()


if __name__ == '__main__':
    sys.exit(main())
