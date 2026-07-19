---
name: html-tutorial
description: Generate polished single-file HTML tutorials with MathJax formulas, code examples, humorous tone, and light/dark theme toggle from any topic or source document.
---

# HTML Tutorial Generator

Generate high-quality, single-file HTML tutorial pages from a topic, code file, or source document. Output is a self-contained HTML with embedded CSS/JS, ready to open in any browser.

## When to use

- User asks for an HTML tutorial/explanation of a technical topic
- User provides code and wants a visual walkthrough
- User has a markdown source and wants an HTML educational page
- User asks to "用HTML详细讲解" or "构建课件"

## Input

- **Topic or source file** (required): the subject to teach, or a code/markdown file to explain
- **Output directory** (optional): where to save the HTML. Default: current directory
- **Number of files** (optional): single file or multi-chapter (e.g., 6 chapters + index)
- **Language** (optional): Chinese (default) or English
- **Theme** (optional): light (default) or dark, or both with toggle

## Procedure

### Step 1: Analyze source material

1. Read the source file(s) if provided
2. Identify the core concepts, formulas, code patterns, and teaching sequence
3. Plan the page structure: sections, code blocks, formula blocks, diagrams
4. For multi-chapter work: create an outline first, then parallelize chapter generation with subagents

### Step 2: Generate HTML with this exact template structure

Every HTML file MUST include these elements in this order:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{TITLE} — {SUBTITLE}</title>
    <!-- MathJax (only if formulas needed) -->
    <script>
        MathJax = {
            tex: { inlineMath: [['$','$'],['\\(','\\)']], displayMath: [['$$','$$'],['\\[','\\]']] },
            svg: { fontCache: 'global' }
        };
    </script>
    <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    <style>
        /* CSS here — see design tokens below */
    </style>
</head>
<body>
    <div class="container">
        <!-- Hero section -->
        <!-- Content sections with .card class -->
        <!-- Code blocks, formula boxes, comparison panels -->
    </div>
</body>
</html>
```

### Step 3: Apply the design system

#### Color Tokens (Light Theme — default)

```css
:root {
    --bg: #fffdf7;
    --card-bg: #ffffff;
    --text: #3a3226;
    --accent: #6c5ce7;
    --accent-light: #a29bfe;
    --accent-good: #00b894;
    --accent-bad: #e17055;
    --border: #e0d5c7;
    --shadow: 0 4px 20px rgba(0,0,0,0.06);
    --radius: 20px;
    --code-bg: #fdf6e3;
    --highlight: #ffeaa7;
}
```

#### Layout

- `max-width: 950px` centered container
- `.card` with `border-radius: 20px`, `box-shadow`, white background
- `.hero` section with dashed border accent, centered title
- Section spacing: `gap: 28px` between cards

#### Code Blocks

```css
.code-block {
    background: var(--code-bg);
    border-radius: 12px;
    padding: 20px;
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 14px;
    line-height: 1.6;
    overflow-x: auto;
    border-left: 4px solid var(--accent);
}
.code-block .comment { color: #93a1a1; }
.code-block .keyword { color: #268bd2; }
.code-block .string { color: #2aa198; }
.code-block .number { color: #d33682; }
```

**Critical**: Use `<pre><code class="code-block">` for code, NOT `formula-box` or `<div>` with inline styles. Apply `.syntax-*` classes for highlighting.

#### Formula Boxes

```css
.formula-box {
    background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    border: 1px solid var(--accent-light);
    margin: 16px 0;
}
```

Use MathJax `$...$` for inline and `$$...$$` for display formulas. Never use Unicode math symbols (π, θ, Σ) as substitutes.

#### Comparison Panels (for before/after, good/bad)

```css
.compare { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.compare-bad { border: 2px solid var(--accent-bad); border-radius: 12px; padding: 16px; }
.compare-good { border: 2px solid var(--accent-good); border-radius: 12px; padding: 16px; }
```

### Step 4: Teaching style

Follow the **Feynman method** as recorded in MEMORY.md:

1. **Intuition first**: Start with a real-world analogy or story
2. **Build to math**: Introduce formulas with explanation of each symbol
3. **Concrete examples**: Code snippets, step-by-step walkthroughs
4. **Humor**: Use 风趣幽默 tone, witty analogies, self-deprecating jokes
5. **Progressive difficulty**: Simple → complex within each section

### Step 5: Quality checklist (before finishing)

Verify ALL of these:

- [ ] MathJax loads correctly: `<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">` is present
- [ ] All math uses `$...$` or `$$...$$` LaTeX syntax, not Unicode symbols
- [ ] Code blocks use `<pre><code class="code-block">` with syntax highlight classes
- [ ] Code blocks are NOT inside `.formula-box` divs
- [ ] CSS has no duplicate class definitions
- [ ] Page renders correctly at 950px max-width
- [ ] All internal links work (if multi-page)
- [ ] Chinese text displays correctly (no encoding issues)
- [ ] Responsive: works on mobile (add `<meta name="viewport">` with `width=device-width`)
- [ ] Single file: all CSS/JS embedded, no external dependencies except MathJax CDN

### Step 6: Multi-chapter output (when applicable)

For 6+ chapters:
1. Generate `index.html` as hub page with links to all chapters
2. Each chapter in its own folder: `NN_chapter_name/index.html`
3. Use parallel subagents (via `actor` tool) for chapter generation
4. All chapters share identical `<head>` pattern (same MathJax config, same CSS tokens)

### Step 7: Theme toggle (optional, when user requests both themes)

Use the proven pattern from minimind3 tutorials:
- `style.css` for dark theme, `light.css` for light theme
- `theme.js` for toggle logic with `localStorage` + system preference detection
- HTML uses `<link id="theme-css" rel="stylesheet" href="style.css">` so JS can swap `href`
- Add toggle button in a fixed position (bottom-right corner)

## Output

- Single HTML file (or folder structure for multi-chapter)
- Self-contained: opens in any browser with internet (for MathJax CDN)
- No build step required

## Anti-patterns to avoid

1. **NEVER use Unicode math** (π, θ, Σ, ε) instead of MathJax LaTeX — breaks on some fonts
2. **NEVER put code inside formula-box** — use dedicated `.code-block` styling
3. **NEVER create markdown-only output** when user asks for HTML — always produce HTML
4. **NEVER skip MathJax script tag** — formulas will render as raw LaTeX text
5. **NEVER use external CSS files** for single-file tutorials — embed everything
6. **NEVER forget `id="theme-css"`** on the link tag if theme toggle may be needed later
