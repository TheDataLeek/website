#!/usr/bin/env python

import pathlib
import subprocess

assets_dir = pathlib.Path(__file__).parent
website_root = assets_dir.parent

# https://stackoverflow.com/questions/653380/how-to-convert-a-pdf-to-png-with-imagemagick-convert-or-ghostscript
resume_cmd = 'gs -sDEVICE=pngalpha -o images/zoefarmer_resume_%01d.png -r144 docs/zoefarmer_resume.pdf'
subprocess.run(resume_cmd, shell=True, cwd=assets_dir)

# Convert presentation and college note PDFs to page images for gallery display.
# Output: {pdf_parent}/{pdf_stem}/page-N.png  (e.g. presentations/chaos/chaos-and-collatz/page-1.png)
search_dirs = [
    website_root / 'presentations',
    website_root / 'college' / 'notes',
]
skip_dirs = {'figure'}

for search_dir in search_dirs:
    for pdf in sorted(search_dir.rglob('*.pdf')):
        if skip_dirs.intersection(pdf.parts):
            continue

        out_dir = pdf.parent / pdf.stem
        out_dir.mkdir(exist_ok=True)

        subprocess.run(
            ['gs', '-sDEVICE=pngalpha', f'-o{out_dir}/page-%d.png', '-r200', str(pdf)],
            capture_output=True,
        )

        pages = len(list(out_dir.glob('page-*.png')))
        print(f'{pdf.relative_to(website_root)}  →  {pages} pages')