#!/usr/bin/env python3
"""
Transform a raw marimo export to Jekyll-compatible markdown (body only).
Usage: python3 convert.py <input.md> <output.md>
"""
import re, sys

COLLAPSE_THRESHOLD = 30


def process_blocks(text):
    lines = text.split('\n')
    out = []
    i = 0

    while i < len(lines):
        line = lines[i]
        m = re.match(r'^```python \{\.marimo(.*?)\}$', line)
        if not m:
            out.append(line)
            i += 1
            continue

        # Find closing fence
        j = i + 1
        while j < len(lines) and lines[j] != '```':
            j += 1

        block = lines[i+1:j]
        body = '\n'.join(block)

        # Drop empty blocks (marimo trailing artifacts)
        if not body.strip():
            i = j + 1
            continue

        # mo.mermaid("""...""") → ```mermaid block
        mm = re.search(r'mo\.mermaid\("""\n(.*?)\n"""\)', body, re.DOTALL)
        if mm:
            out.append('```mermaid')
            out.append(mm.group(1))
            out.append('```')
            i = j + 1
            continue

        # Regular/hidden code block
        out.append('```python')
        out.extend(block)
        out.append('```')

        # If block calls mo.image(), append image reference
        if 'mo.image(' in body:
            png = re.search(
                r'notebook_dir\(\)\s*/\s*["\']([^"\']+\.png)["\']', body
            )
            if png:
                if len(block) >= COLLAPSE_THRESHOLD:
                    out.append('{: .code-collapsed}')
                out.append(f'![png]({png.group(1)})')

        i = j + 1

    return '\n'.join(out)


def merge_adjacent_python_blocks(text):
    """Combine consecutive ```python fences with nothing but blank lines between
    them (no prose, no image) into a single fenced block."""
    lines = text.split('\n')
    out = []
    i = 0
    while i < len(lines):
        if lines[i] != '```python':
            out.append(lines[i])
            i += 1
            continue

        # Collect this block's body
        j = i + 1
        body = []
        while j < len(lines) and lines[j] != '```':
            body.append(lines[j])
            j += 1
        j += 1  # past closing fence

        # Keep absorbing directly-adjacent python blocks
        while True:
            k = j
            while k < len(lines) and lines[k].strip() == '':
                k += 1
            if k < len(lines) and lines[k] == '```python':
                body.append('')
                k += 1
                while k < len(lines) and lines[k] != '```':
                    body.append(lines[k])
                    k += 1
                j = k + 1
            else:
                break

        out.append('```python')
        out.extend(body)
        out.append('```')
        i = j

    return '\n'.join(out)


def fix_math(text):
    # Protect existing $$ so the single-$ pass doesn't double them again
    text = text.replace('$$', '\x00')
    text = re.sub(r'\$([^$\n\x00]+?)\$', r'$$\1$$', text)
    return text.replace('\x00', '$$')


def convert(src):
    # Strip marimo frontmatter block
    text = re.sub(r'^---\n.*?\n---\n\n?', '', src, flags=re.DOTALL)
    text = process_blocks(text)
    text = merge_adjacent_python_blocks(text)
    text = fix_math(text)
    # Collapse runs of 3+ blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


if __name__ == '__main__':
    inp, out = sys.argv[1], sys.argv[2]
    with open(inp) as f:
        src = f.read()
    result = convert(src)
    with open(out, 'w') as f:
        f.write(result + '\n')
    print(f'Written: {out}')
