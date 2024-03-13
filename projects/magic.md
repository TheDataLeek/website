---
layout: post
nav-menu: false
show_tile: false
title: "Creating 'Magic' Circles with Javascript"
---

<div id='output'>
</div>

<script>
function addCircle(drawing, radius) {
    return drawing.circle(100)
        .attr({
            cx: size / 2,
            cy: size / 2,
            r: radius
        })
        .fill({
            color: '#000'
            opacity: 0,
        })
        .stroke({
            color: '#fff'
            opacity: 1,
            width: 2,
        });
}

const size = 500;
const unicodeRange = ['10600', '1077F'];
var draw = SVG().addTo('#output').size(size, size);
addCircle(draw, 1)
addCircle(draw, 1)
addCircle(draw, 2)
addCircle(draw, 3)
addCircle(draw, 5)
addCircle(draw, 8)
addCircle(draw, 13)
</script>