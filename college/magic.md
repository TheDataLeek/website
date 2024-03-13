---
layout: post
nav-menu: false
show_tile: false
title: "Creating 'Magic' Circles with Javascript"
---

<div id='output'>
</div>

<script>
const size = 500;
var draw = SVG().addTo('#output').size(size, size);
draw.circle(100).attr({cx: size / 2, cy: size / 2})
</script>