#!/usr/bin/env python

import functools
import http.server
import pathlib
import subprocess
import threading

from cyclopts import App

from playwright.sync_api import sync_playwright

app = App()

assets_dir = pathlib.Path(__file__).parent
website_root = assets_dir.parent

@app.default
def all():
    resume()
    pdfs()
    reveal()

@app.command()
def resume():
    # https://stackoverflow.com/questions/653380/how-to-convert-a-pdf-to-png-with-imagemagick-convert-or-ghostscript
    resume_cmd = 'gs -sDEVICE=pngalpha -o images/zoefarmer_resume_%01d.png -r144 docs/zoefarmer_resume.pdf'
    subprocess.run(resume_cmd, shell=True, cwd=assets_dir)


@app.command()
def pdfs():
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

@app.command()
def reveal():
    # Screenshot Reveal.js presentations slide-by-slide using Playwright.
    # Output: {html_parent}/{stem}/page-N.png  (e.g. presentations/evolve/evolve_interview/page-1.png)
    reveal_presentations = [
        {
            'html': website_root / 'presentations' / 'evolve' / 'evolve_interview.slides.html',
            'out_dir': website_root / 'presentations' / 'evolve' / 'evolve_interview',
        },
        {
            'html': website_root / 'presentations' / 'd3reuse' / 'D3Reuse.slides.html',
            'out_dir': website_root / 'presentations' / 'd3reuse' / 'D3Reuse',
        },
        {
            'html': website_root / 'presentations' / 'politicalboundaries' / 'politicalboundaries.slides.html',
            'out_dir': website_root / 'presentations' / 'politicalboundaries' / 'politicalboundaries',
        },
        {
            'html': website_root / 'presentations' / 'speedsnakes' / 'snakes.html',
            'out_dir': website_root / 'presentations' / 'speedsnakes' / 'snakes',
        },
    ]

    PORT = 18765
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(website_root))
    server = http.server.HTTPServer(('127.0.0.1', PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    with sync_playwright() as p:
        browser = p.chromium.launch()

        for pres in reveal_presentations:
            html_path = pres['html']
            out_dir = pres['out_dir']
            out_dir.mkdir(exist_ok=True)

            rel = html_path.relative_to(website_root)
            url = f'http://127.0.0.1:{PORT}/{rel}'

            page = browser.new_page(viewport={'width': 960, 'height': 700})
            page.goto(url, wait_until='networkidle')
            page.wait_for_function("() => typeof Reveal !== 'undefined' && Reveal.isReady()", timeout=15000)
            page.evaluate("Reveal.configure({transition: 'none', transitionSpeed: 'default', margin: 0})")

            slides_info = page.evaluate("""
                () => Reveal.getSlides().map(function(s) {
                    var idx = Reveal.getIndices(s);
                    return {h: idx.h, v: idx.v === undefined ? 0 : idx.v};
                })
            """)

            for i, idx in enumerate(slides_info, start=1):
                page.evaluate(f"Reveal.slide({idx['h']}, {idx['v']})")
                page.wait_for_timeout(50)
                page.screenshot(path=str(out_dir / f'page-{i}.png'), clip={'x': 0, 'y': 0, 'width': 960, 'height': 700})

            page.close()
            print(f'{rel}  →  {len(slides_info)} slides')

        browser.close()

    server.shutdown()

if __name__ == '__main__':
    app()