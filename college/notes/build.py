#!/usr/bin/env python

import sys
import re
import pathlib
import subprocess


def main():
    root = pathlib.Path()
    paths = []
    built_files = False
    for path in root.glob('*'):
        if path.is_dir():
            paths.append(path)
            built_files &= build_latex(path)
            build_index(path)
    if built_files:
        remove_buildfiles(root)
    build_root_index(root, paths)

def build_latex(path):
    """
    Only build if there are no pdf files in the directory
    """
    print(f"Building notes for {path}\n")
    pdf_files = len(list(path.glob('*.pdf'))) != 0
    if not pdf_files:
        subprocess.run("make", shell=True, cwd=path)
        return True
    return False

def build_index(path):
    dir_name = split_camel_case(path.stem)
    template = f"""---
layout: post
title: {dir_name} Notes
nav-menu: false
show_tile: false
---

"""
    note_files = sorted(list(path.glob('*.pdf')), key=lambda x: x.name)
    num_files = len(note_files)
    for note_file in note_files:
        filename = note_file.stem
        note_name = split_camel_case(filename)
        if num_files > 1:
            template += f'\n# {note_name}\n\n'
        template += f"""
<iframe src="/college/notes/{path.name}/{note_file.name}"
        style="width: 100%; height: 40em;">
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


def build_root_index(root, notes):
    template = f"""---
layout: post
title: Notes
nav-menu: false
show_tile: false
---
"""
    notes.sort(key=lambda x: x.name)
    for note in notes:
        name = split_camel_case(note.stem)
        template += f"""
* [{name}]({note.name})
"""
    (root / 'index.md').write_text(template)

def split_camel_case(word):
    return re.sub('([A-Z0-9])', r' \1', word).strip()


if __name__ == '__main__':
    sys.exit(main())
