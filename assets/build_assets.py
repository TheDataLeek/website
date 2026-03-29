#!/usr/bin/env python

import functools
import http.server
import pathlib
import subprocess
import threading

from cyclopts import App

from playwright.sync_api import sync_playwright

app = App()

ASSETS = pathlib.Path(__file__).parent
ROOT = ASSETS.parent

@app.default
def all():
    resume()
    pdfs()

@app.command()
def resume():
    # https://stackoverflow.com/questions/653380/how-to-convert-a-pdf-to-png-with-imagemagick-convert-or-ghostscript
    resume_cmd = 'gs -sDEVICE=pngalpha -o images/zoefarmer_resume_%01d.png -r144 docs/zoefarmer_resume.pdf'
    subprocess.run(resume_cmd, shell=True, cwd=ASSETS)


@app.command()
def pdfs():
    # Convert presentation and college note PDFs to page images for gallery display.
    # Output: {pdf_parent}/{pdf_stem}/page-N.png  (e.g. presentations/chaos/chaos-and-collatz/page-1.png)
    search_dirs = [
        ROOT / 'presentations',
        ROOT / 'college' / 'notes',
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
            print(f'{pdf.relative_to(ROOT)}  →  {pages} pages')

if __name__ == '__main__':
    app()