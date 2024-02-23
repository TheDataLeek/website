---
title: "Reusable Charting with D3"
date: "2018-04-29"
categories: 
  - "d3"
tags: 
  - "d3"
  - "data"
  - "framework"
  - "javascript"
  - "js"
  - "visualization"
---

I love the [D3js framework](https://d3js.org/), however it's not initially intuitive how to create charts that have reusable components. After working extensively with this framework over the last few months, I've found a good way to introduce a little more extensibility into the default approach.

For the following examples we use ES6 features, along with flexbox for positioning, which means that it might not work on all browsers. In production, I would recommend transpiling to ES5 for compatibility.

# Motivation

D3 does not inherently provide many tools to help developers create reusable charts. Often times documentation can be sparse on library features, and most code examples are obtuse and hard to follow (or use) unless you're an expert in the framework.

The goal of this extension is to allow for more use and reuse of these charts.

# The core framework

There's a lot of code here that at first glance doesn't seem that related to D3, but skim through and I'll explain below.

class Chart {
    constructor(selector, params={}) {
        // selector is just the selection string
        this.selector = selector;
        // svg is the d3 svg element
        this.svg = d3.select(selector).append('svg');
        // chart is the main group that we use for everything
        this.chart = this.svg.append('g');
        // margins are the empty space outside the chart
        this.margin = params.margin || {
            top: 0,
            bottom: 0,
            left: 0,
            right: 0,
        };
    }

    get selected() {
        // getter for the jquery selector
        return $(this.selector);
    }

    draw() {
        // this is overridden by subclass
    }

    newGroup(name, parent=undefined) {
        if (parent === undefined) {
            this.chart.selectAll(\`.${name}\`).remove();
            this\[name\] = this.chart.append('g').classed(name, true);
        } else {
            parent.selectAll(\`.${name}\`).remove();
            parent\[name\] = parent.append('g').classed(name, true);
        }
    }
}

In short this attempts to provide a universal superclass for all charts created with D3. This allows us to solve universal problems (or even common problems) only once, include the solution in this superclass, and never have to solve the same problem again. For instance, this initial superclass adds the following features to _any_ plot initialized with it

- Margins
- Easy-to-initialize elements

# Basic Demonstration

Using this framework we can plot a simple resizing rectangle.

With corresponding code,

class EmptyRectangle extends Chart {
    constructor(selector, params={}) {
        super(selector, params);

        this.init();
    }

    draw() {
        this.newGroup('background');

        this.background.append('rect')
            .attr('width', this.width)
            .attr('height', this.height)
            .attr('fill', 'steelblue');
    }
}

const rect = new EmptyRectangle('.resizable-rectangle');

# Something more complicated

Showing more power of this approach, we can create a simple margin demonstration

With corresponding code,

class MarginDemonstration extends Chart {
    constructor(selector, params={}) {
        super(selector, params);
        this.init();
    }

    drawContainer() {
        this.newGroup('containerRect');

        const strokeWidth = 2;

        this.containerRect
            .attr(
                'transform',
                \`translate(
                    ${-this.margin.left + strokeWidth},
                    ${-this.margin.top + strokeWidth}
                )\`
            )
            .append('rect')
            .attr('width', this.containerWidth - (2 \* strokeWidth))
            .attr('height', this.containerHeight - (2 \* strokeWidth))
            .attr('fill', 'none')
            .attr('stroke', 'black')
            .attr('stroke-width', strokeWidth)
            .attr('stroke-dasharray', ('3, 3'));
    }

    drawChart() {
        this.newGroup('chartRect');

        this.chartRect
            .append('rect')
            .attr('width', this.width)
            .attr('height', this.height)
            .attr('fill', 'none')
            .attr('stroke', 'black')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', ('3, 3'));
    }

    drawLabels() {
        this.newGroup('labels');

        this.labels.append('text')
            .attr('transform', \`translate(${this.width / 2}, ${this.height / 2})\`)
            .attr('fill', 'black')
            .style('text-anchor', 'middle')
            .style('vertical-align', 'baseline')
            .text('this.chart');

        this.labels.append('text')
            .attr('transform', \`translate(${this.width / 2}, ${this.height - 3})\`)
            .attr('fill', 'black')
            .style('text-anchor', 'middle')
            .style('vertical-align', 'baseline')
            .text('this.width');

        this.labels.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -this.height / 2)
            .attr('y', 12)
            .attr('fill', 'black')
            .style('text-anchor', 'middle')
            .style('vertical-align', 'baseline')
            .text('this.height');
    }

    draw() {
        this.drawContainer();
        this.drawChart();
        this.drawLabels();
    }
}

const margins = new MarginDemonstration(
    '.margins-example',
    {
        margin: {
            top: 50,
            bottom: 50,
            left: 50,
            right: 50,
        },
    }
)

# So what?

If you're creating a ton of D3 plots, I highly recommend tinkering with plot creation using an approach like this. It's saved me a ton of time, and helped improve _every_ part of working with D3.

Feel free to download and play with the code at [https://www.dataleek.io/js/ex.js](https://www.dataleek.io/js/ex.js)

<script src="https://code.jquery.com/jquery-3.3.1.min.js" integrity="sha256-FgpCb/KJQlLNfOu91ta32o/NMZxltwRo8QtmkMRdAu8=" crossorigin="anonymous"></script>

<script src="https://d3js.org/d3.v5.min.js"></script>

<script src="https://www.dataleek.io/js/ex.js"></script>
