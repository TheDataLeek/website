// Used https://bl.ocks.org/john-guerra/43c7656821069d00dcbc for reference

// set initial parameters
var width = 900,
    height = 600,
    graphHeight = 300;

// COLORS
var colours = {
    // default_color: 'rgb(104, 150, 137, 0.5)',
    default_color: 'rgb(104, 150, 137)',
    null_colour: 'rgb(255, 255, 255)',
    // selected_color: 'rgb(114, 87, 82)'
    selected_color: 'rgb(148, 148, 146)'
}

var colourRange = d3.scaleLinear()
    .domain([50, 300])
    .range(['rgb(252,251,253)', 'rgb(63,0,125)']);

var map_colours;

var desc_columns = [
    'description',
    'price'
]

var descriptions;

// VARIABLE INITIALIZATION
// map projection
var projection = d3.geoMercator()
    .scale(140000)  // zoom waaay in 
    .center([-71.0589, 42.31])  // center on boston center [lat, long]
    .translate([width / 2, height / 2]);  // scale to dimensions

// Set svg width/height
// map
var svg = d3.select('#map')
    .attr('width', width)
    .attr('height', height);

// num listings bars
var svg2 = d3.select('#graph')
    .attr('width', width)
    .attr('height', graphHeight);

// hist
var svg3 = d3.select('#hist')
    .attr('width', width)
    .attr('height', graphHeight);

// Add background
// map
svg.append('rect')
    .attr('class', 'background')
    .attr('width', width)
    .attr('height', height);

// graph
svg2.append('rect')
    .attr('class', 'background')
    .attr('width', width)
    .attr('height', graphHeight);

// hist
svg3.append('rect')
    .attr('class', 'background')
    .attr('width', width)
    .attr('height', graphHeight);

var path = d3.geoPath(projection);

// new svg group
var g = svg.append('g');

var mapLayer = g.append('g')
    .classed('map-layer', true);

var pointLayer = g.append('g')
    .classed('point-layer', true);

var textLayer = g.append('g')
    .classed('text-layer', true);

var x = d3.scaleBand().rangeRound([0, 0.9 * width]).padding(0.1),
    y = d3.scaleLinear().rangeRound([graphHeight / 2, 0]);

var g2 = svg2.append('g')
    .attr('transform', 'translate(40, 10)');

var graphLayer = g2.append('g')
    .classed('graph-layer', true);

var g3 = svg3.append('g')
    .attr('transform', 'translate(40, -10)');

var histLayer = g3.append('g')
    .classed('hist-layer', true);

var histTextLayer = g3.append('g')
    .classed('hist-text-layer', true);

d3.json('./airbnb.geojson', function(mapData) {
    var features = mapData.features;

    // D3's groupby
    var neighbourhood_data = d3.nest()
        .key(function(d) { return d.properties["neighbourhood_cleansed"]; })
        .rollup(function(v) { return {
            values: v,
            avgPrice: d3.mean(v, function(e) {
                return e.properties.price.slice(1, e.properties.price.length);
            })
        } })
        .sortKeys(d3.ascending)
        .entries(features);

    // MAP COLOURS
    map_colours = neighbourhood_data.map(function(d) {
        return {
            key: d.key,
            avgPrice: d.value.avgPrice,
            colour: colourRange(d.value.avgPrice)
        }
    });

    // DRAW INDIVIDUAL LISTINGS
    pointLayer.selectAll('circle')
        .data(features)
        .enter()
        .append('circle')
        .attr('cx', function(d) {
            return projection(d.geometry.coordinates)[0]; })
        .attr('cy', function(d) {
            return projection(d.geometry.coordinates)[1]; })
        .attr('r', '1px')
        .attr('fill', 'red');

    // BAR VIZ
    // bar chart reference https://bl.ocks.org/mbostock/3885304

    x.domain(neighbourhood_data.map(function(d) { return d.key; }));
    y.domain([0, d3.max(neighbourhood_data, function(d) { return d.value.values.length; })]);

    graphLayer.append('g')
        .attr('class', 'axis axis--x')
        .attr('transform', 'translate(0,' + graphHeight / 2 + ')')
        .call(d3.axisBottom(x))
        .selectAll('text')
        .attr('dx', '1em')
        .style('text-anchor', 'start')
        .attr('transform', 'rotate(45)');

    graphLayer.append('g')
        .attr('class', 'axis axis--y')
        .call(d3.axisLeft(y).ticks(10))
        .append('text')
        .attr('y', 6)
        .attr('dy', '0.71em')
        .attr('text-anchor', 'end')
        .text('Number of Listings');

    graphLayer.selectAll('.bar')
        .data(neighbourhood_data)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('value', function(d) { return d.key; })
        .attr('num', function(d) { return d.value.values.length })
        .style('fill', colours.default_color)
        .attr('fill-opacity', 0.5)
        .attr('x', function(d) { return x(d.key); })
        .attr('y', function(d) { return y(d.value.values.length); })
        .attr('width', x.bandwidth())
        .attr('height', function(d) { return (graphHeight / 2) - y(d.value.values.length); })
        .on('click', clicked)
        .on('mouseover', mouseover)
        .on('mouseout', mouseout);

    // add bar graph label
    graphLayer.append('text')
        .text('Number of Listings')
        .attr('x', 0.2 * width)
        .attr('y', 10)

    function mapColour(d) {
        var name = d.properties.Name;
        var colour = map_colours.filter(function(d) {
            return d.key == name;
        });
        if (colour.length == 0) {
            return colours.null_colour;
        } else {
            return colour[0].colour;
        }
    }

    function mouseout(d) {
        mapLayer.selectAll('path')
            .style('fill', mapColour);

        graphLayer.selectAll('rect')
            .style('fill', colours.default_color);

        textLayer.selectAll('text').remove();
    }

    function clicked(d) {
        // get name of neighborhood
        var name = d.key || d.properties.Name;

        // Tabulate
        tabulate_neighborhood(name);

        histogram(name);
    }

    function mouseover(d) {
        // make it so it uses either name based on graph vs map mouseover
        var name = d.key || d.properties.Name;

        var district = mapLayer.select('[value="' + name + '"]');

        graphLayer.select('[value="' + name + '"]')
            .style('fill', colours.selected_color);

        district.style('fill', colours.selected_color);

        // Add label
        textLayer.append('text')
            .text(name)
            .attr('x', 0.6 * width)
            .attr('y', 0.8 * height);

    }

    // MAP VIZ
    // Boston GeoJSON data from https://data.boston.gov/dataset/boston-neighborhoods
    d3.json('./Boston_Neighborhoods.geojson', function(error, mapData) {
        var features = mapData.features;

        // draw each district
        mapLayer.selectAll('path')
            .data(features)
            .enter()
            .append('path')
            .attr('d', path)  // path descriptions
            .attr('vector-effect', 'non-scaling-stroke')  // prevents distortions
            .attr('value', function(d) { return d.properties.Name; })
            .style('fill', mapColour)
            .attr('fill-opacity', 0.5)
            .on('click', clicked)
            .on('mouseover', mouseover)
            .on('mouseout', mouseout);
    });

    // Add map name
    mapLayer.append('text')
        .text('Boston\'s Neighbourhoods with AirBnB Listings')
        .style('fill', 'black')
        .attr('x', 0.01 * width)
        .attr('y', 0.05 * height);

    // SCALE FOR MAP COLOURS
    var defs = svg.append('defs');

    var gradient = defs.append('linearGradient')
        .attr('id', 'linear-gradient');

    gradient.attr('x1', '0%')
        .attr('y1', '0%')
        .attr('x2', '100%')
        .attr('y2', '0%');

    // start color
    gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', 'rgb(252,251,253)')
        .attr('fill-opacity', 0.5);

    // stop color
    gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', 'rgb(63,0,125)')
        .attr('fill-opacity', 0.5);

    svg.append('rect')
        .attr('x', 0.7 * width)
        .attr('y', 0.9 * height)
        .attr('width', 200)
        .attr('height', 20)
        .style('fill', 'url(#linear-gradient)');

    svg.append('text')
        .text('$50')
        .attr('x', 0.7 * width)
        .attr('y', 0.9 * height - 10);

    svg.append('text')
        .text('$300')
        .attr('x', 0.7 * width + 175)
        .attr('y', 0.9 * height - 10);


    // DESCRIPTIONS
    descriptions = neighbourhood_data.map(function(d) {
        return {
            key: d.key,
            info: d.value.values.map(function(e) {
                return {
                    summary: e.properties.summary,
                    description: e.properties.description,
                    price: parseFloat(
                        e.properties.price.slice(1, e.properties.price.length)
                    ),
                };
            }).sort(function(a, b) {
                return d3.ascending(a.price, b.price);
            })
        };
    });

    // HISTOGRAM
    function histogram(name) {
        histLayer.remove();

        var y_offset = 25;

        histLayer = g3.append('g')
            .classed('hist-layer', true);

        var histx = d3.scaleLinear()
            .domain([0, 300])
            .rangeRound([0, 0.75 * width]);

        var data = neighbourhood_data.filter(function(d) {
            return d.key == name;
        });

        if (data.length == 0) {
            histLayer.append('text')
                .text('No data for ' + name)
                .attr('x', 0.1 * width)
                .attr('y', 30)
            return;
        }

        data = data[0].value.values;

        data = data.map(function(d) {
            return parseFloat(d.properties.price.slice(1, d.properties.price.length));
        });

        var bins = d3.histogram()
            .domain(histx.domain())
            .thresholds(histx.ticks(5))
            (data);

        var histy = d3.scaleLinear()
            .domain([0, 0.6])
            .range([0.8 * graphHeight, 0]);

        var bar = histLayer.selectAll('.bar')
            .data(bins)
            .enter()
            .append('g')
            .attr('class', 'bar')
            .style('fill', colours.default_color)
            .attr('fill-opacity', 0.5)
            .attr('value', function(d) { return d.length / data.length})
            .attr('transform', function(d) {
                return 'translate(' + histx(d.x0) + ',' + histy(d.length / data.length) + ')';
            }).on('mouseover', function(d) {
                histTextLayer.append('text')
                    .style('fill', 'black')
                    .attr('fill-opacity', 1)
                    .attr('x', 0.7 * width)
                    .attr('y', 0.1 * height)
                    .text(d3.select(this).attr('value'));
            }).on('mouseout', function(d) {
                histTextLayer.selectAll('text').remove();
            });

        var barWidth = histx(bins[0].x1) - histx(bins[0].x0);

        bar.append('rect')
            .attr('x', 1)
            .attr('y', y_offset)
            .attr('width', barWidth - 1)
            .attr('height', function(d) { return (0.8 * graphHeight) - histy(d.length / data.length)})

        histLayer.append('g')
            .attr('class', 'axis axis--x')
            .attr('transform', 'translate(0,' + ((0.8 * graphHeight) + y_offset) + ')')
            .call(d3.axisBottom(histx));

        histLayer.append('g')
            .attr('class', 'axis axis--y')
            .attr('transform', 'translate(0,' + ((0.8 * graphHeight) - 215) + ')')
            .call(d3.axisLeft(histy).ticks(10))
            .append('text')
            .attr('dy', '0.71em')
            .attr('text-anchor', 'end');

        // add hist graph label
        histLayer.append('text')
            .text('Normalized Histogram of Prices for ' + name)
            .attr('x', 0.1 * width)
            .attr('y', 30)
    }

});

// CHART VIZ
// http://bl.ocks.org/jfreels/6734025
function tabulate_neighborhood(name) {
    d3.select('#chart').html('');

    var table = d3.select('#chart')
        .append('table')
        .attr('border', 1);
    var thead = table.append('thead');
    var tbody = table.append('tbody');

    neighbourhood_data = descriptions.filter(function(d) {
        return d.key == name;
    })[0];

    if (neighbourhood_data == null) {
        return;
    }

    // header
    thead.append('tr')
        .selectAll('th')
        .data(desc_columns)
        .enter()
        .append('th')
        .text(function (col_name) { return col_name });

    // rows
    var rows = tbody.selectAll('tr')
        .data(neighbourhood_data.info)
        .enter()
        .append('tr');

    // cells
    var cells = rows.selectAll('td')
        .data(function(row) {
            var rowdata = desc_columns.map(function(col) {
                return {
                    column: col,
                    value: row[col]
                }
            })
            return rowdata;
        }).enter()
        .append('td')
        .text(function(d) {
            if (d.column == 'price') {
                return '$' + d.value;
            } else {
                return d.value;
            }
        });
}

