#!/usr/bin/env python3
"""
Build Script for Gomoku Web App

Merges index.html, styles.css, and JS modules into a single standalone HTML file
for CDN deployment.

Usage:
    python3 build.py
"""

import re
from pathlib import Path


def inline_css(html_content: str) -> str:
    """Replace CSS link with inline style tag."""
    with open('styles.css', 'r', encoding='utf-8') as f:
        css_content = f.read()

    # Replace <link rel="stylesheet" href="styles.css"> with <style>...</style>
    replacement = f'<style>\n{css_content}\n    </style>'
    html_content = re.sub(
        r'<link rel="stylesheet" href="styles\.css">',
        replacement,
        html_content
    )

    return html_content


def inline_js(html_content: str) -> str:
    """Replace JS script tags with inline scripts."""
    def replace_script(match):
        filename = match.group(1)
        with open(f'js/{filename}', 'r', encoding='utf-8') as f:
            js_content = f.read()
        return f'<script>\n{js_content}\n    </script>'

    # Replace <script src="js/..."></script> with <script>...</script>
    html_content = re.sub(
        r'<script src="js/(.*?)"></script>',
        replace_script,
        html_content
    )

    return html_content


def build():
    """Main build function."""
    print('Building gomoku-standalone.html...')

    # Read the development HTML
    with open('index.html', 'r', encoding='utf-8') as f:
        html = f.read()

    # Inline CSS
    print('  Inlining CSS...')
    html = inline_css(html)

    # Inline JS modules
    print('  Inlining JS modules...')
    html = inline_js(html)

    # Write standalone version
    with open('gomoku-standalone.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print('✓ Built gomoku-standalone.html successfully!')

    # Print file sizes
    standalone_size = Path('gomoku-standalone.html').stat().st_size
    print(f'  Standalone file size: {standalone_size:,} bytes ({standalone_size / 1024:.1f} KB)')


if __name__ == '__main__':
    build()
