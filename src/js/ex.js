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

    init() {
        // need to initialize chart initially
        this.resize();
        // and on resize, redraw
        $(window).resize(() => {
            this.resize();
        })
    }

    draw() {
        // this is overridden by subclass
    }

    resize() {
        // calculates new dimensions and draws
        // https://bl.ocks.org/curran/3a68b0c81991e2e94b19
        this.containerWidth = this.selected.width();
        this.containerHeight = this.selected.height();

        this.width = this.containerWidth - this.margin.left - this.margin.right;
        this.height = this.containerHeight - this.margin.top - this.margin.bottom;

        this.svg
            .attr('width', this.containerWidth)
            .attr('height', this.containerHeight);

        this.chart
            .attr('transform', `translate(${this.margin.left}, ${this.margin.top})`);

        this.draw();
    }

    newGroup(name, parent=undefined) {
        if (parent === undefined) {
            this.chart.selectAll(`.${name}`).remove();
            this[name] = this.chart.append('g').classed(name, true);
        } else {
            parent.selectAll(`.${name}`).remove();
            parent[name] = parent.append('g').classed(name, true);
        }
    }
}

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
                `translate(
                    ${-this.margin.left + strokeWidth},
                    ${-this.margin.top + strokeWidth}
                )`
            )
            .append('rect')
            .attr('width', this.containerWidth - (2 * strokeWidth))
            .attr('height', this.containerHeight - (2 * strokeWidth))
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
            .attr('transform', `translate(${this.width / 2}, ${this.height / 2})`)
            .attr('fill', 'black')
            .style('text-anchor', 'middle')
            .style('vertical-align', 'baseline')
            .text('this.chart');

        this.labels.append('text')
            .attr('transform', `translate(${this.width / 2}, ${this.height - 3})`)
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

const rect = new EmptyRectangle('.resizable-rectangle');

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


