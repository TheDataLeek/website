---
layout: page
title: Neural Networks
description: Interactive explorations of neural network fundamentals — built from scratch, visualized in depth.
nav-menu: false
---

<div class="college-wrapper">
  <header class="college-header">
    <h1 class="college-header__title">Neural Networks</h1>
    <p class="college-header__desc">Interactive explorations of neural network fundamentals — built from scratch, visualized in depth.</p>
  </header>

  <div class="college-section" style="--college-accent: var(--accent-primary)">
    <div class="college-section__cells">
      <article class="college-cell">
        <a href="{{ '/projects/neural-networks/mlp' | relative_url }}" class="college-cell__inner">
          <h3 class="college-cell__title">MLP Function Approximation</h3>
          <p class="college-cell__desc">Trains a four-layer MLP to approximate x² and visualises the full convergence trajectory, then reframes the trained network as a key-value memory — showing how neurons store receptive regions as keys and gradient-weighted contributions as values.</p>
        </a>
      </article>
      <article class="college-cell">
        <a href="{{ '/projects/neural-networks/attention' | relative_url }}" class="college-cell__inner">
          <h3 class="college-cell__title">Attention as a Soft Lookup Table</h3>
          <p class="college-cell__desc">A minimal model built from scratch uses attention to learn color-to-noun mappings, showing how scaled dot-product attention implements a differentiable soft lookup table — and what the model learns geometrically after training.</p>
        </a>
      </article>
      <article class="college-cell">
        <a href="{{ '/projects/neural-networks/transformer' | relative_url }}" class="college-cell__inner">
          <h3 class="college-cell__title">Character-Level Transformer</h3>
          <p class="college-cell__desc">A single-layer transformer trained on next-character prediction in a cyclic pangram, walking through every architectural component — causal masking, residual connections, layer normalisation — with geometric visualisations of what the model learns.</p>
        </a>
      </article>
    </div>
  </div>
</div>
