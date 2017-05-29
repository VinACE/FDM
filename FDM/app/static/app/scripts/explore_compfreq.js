// explore.js

'use strict';

var chartCircles = function (tag, data) {
    var chart = d3.select(tag);
    // Set the chart height and width from data
    chart.attr('height', data.height).attr('width', data.width);
    // Create some circles using data
    chart.selectAll('circle').data(data.circles)
        .enter()
        .append('circle')
        .attr('cx', function (d) { return d.x })
        .attr('cy', function (d) { return d.y })
        .attr('r', function (d) { return d.r });
};

var circledata = {
    width: 600, height: 300,
    circles: [
        { 'x': 50, 'y': 30, 'r': 20 },
        { 'x': 70, 'y': 80, 'r': 10 },
        { 'x': 160, 'y': 60, 'r': 10 },
        { 'x': 200, 'y': 100, 'r': 5 },
    ]
};

function type(d) {
    d.value = +d.value; // coerce to number
    return d;
}

var bardata = {
    width: 300, height: 200,
    bars: [
        { 'comp': 'Albert Einstein', 'freq': 10 },
        { 'comp': 'V.S. Naipaul', 'freq': 5 },
        { 'comp': 'Curry', 'freq': 11 },
        { 'comp': 'Dorothy Hodgkin', 'freq': 12 }
    ]
};

//chartCircles('#chart1', circledata)


var ingr_freq_barchart1 = function (tag, data) {
    var chart = d3.select(tag);
    var width = 800
    var height = 200
    chart.attr('height', height).attr('width', width);
    var y_max = d3.max(data, function (d) { return +d.freq; });
    var x_max = data.length;
    var yScale = d3.scale.linear()
        .domain([0, y_max])
        .range([height, 0]);
    var xScale = d3.scale.ordinal()
        .domain(d3.range(x_max))
        .rangeRoundBands([0, width], 0.1);
    // Create bars using data
    chart.selectAll("bar")
        .data(data)
        .enter().append("rect")
        .style("stroke", "gray")
        .style("fill", "blue")
        .attr("class", "bar")
        .attr("x", function (d, i) { return i * xScale.rangeBand() })
        .attr("y", function (d) { return yScale(d.freq) })
        .attr("width", xScale.rangeBand())
        .attr("height", function (d) { return height - yScale(d.freq); })
        .on("mouseover", function (d, i) {
            d3.select(this).style("fill", "aliceblue");
            //Get this bar's x/y values, then augment for the tooltip
            var xPosition = parseFloat(d3.select(this).attr("x")) + xScale.rangeBand() / 2;
            var yPosition = parseFloat(d3.select(this).attr("y")) + 14;
            //Create the tooltip label
            chart.append("text")
               .attr("id", "tooltip")
               .attr("x", xPosition)
               .attr("y", yPosition)
               .attr("text-anchor", "middle")
               .attr("font-family", "sans-serif")
               .attr("font-size", "11px")
               .attr("font-weight", "bold")
               .attr("fill", "black")
               .text(d.comp + ": " + d.freq);
        })
        .on("mouseout", function (d, i) {
            d3.select(this).style("fill", "blue");
            //Remove the tooltip
            d3.select("#tooltip").remove();
        });
};


var ingr_bucket_piechart = dc.pieChart('#ingr-bucket-piechart');
var ingr_freq_barchart = dc.barChart('#ingr-freq-barchart2');
var ingr_freq_table = dc.dataTable('#ingr-freq-table');
var ndx = crossfilter();
var comp_dim;

function bucketCount_from_threshold() { 
    var scoreThreshold=document.getElementById('slideRange').value; 
    scoreThreshold=parseFloat(scoreThreshold); 
    if (isNaN(scoreThreshold)) { 
        scoreThreshold=100 
    }
    var bucket_size = scoreThreshold;   
    return ndx.dimension(function (d) { 
        var bucket = Math.ceil(+d.freq / bucket_size);
        d.bucket = ((bucket - 1) * bucket_size).toString() + '-' + (bucket * bucket_size).toString()
    }); 
} 

var bucketCount = bucketCount_from_threshold();

//## change slider score value to re-assign the data in pie chart 
function updateSlider(slideValue) { 
    var sliderDiv = document.getElementById("sliderValue"); 
    sliderDiv.innerHTML = slideValue;
    bucketCount.dispose();
    bucketCount = bucketCount_from_threshold();
    var bucket_dim = ndx.dimension(function (d) { return d.bucket; }),
        nrcomp_bucket = bucket_dim.group().reduceCount();
    ingr_bucket_piechart
        .dimension(bucket_dim)
        .group(nrcomp_bucket)
    dc.redrawAll(); 
} 

function on_filteringr_bucket_piechart()
{
    //    var bucket_comp = ['00011484', '00012461'];
    var bucket_comp = [];
    var filter_comp = comp_dim.top(20);
    for (var i = 0; i < filter_comp.length; i++) {
        bucket_comp[i] = filter_comp[i].comp;
    }
    var barwidht = (bucket_comp.length > 0) ? 300 / bucket_comp.length : 300;
    ingr_freq_barchart
        .x(d3.scale.ordinal().domain(bucket_comp).range(0, bucket_comp.length).rangeRoundBands([0, 300], .1))
//        .xUnits(function (start, end, xDomain) { return xDomain.length; })
//          .xUnits(dc.units.ordinal)
        .xUnits(function () { return bucket_comp.length; })
}

function on_renderletingr_freq_barchart() {
    // rotate x-axis labels
    ingr_freq_barchart.selectAll('g.x text')
        .attr('transform', 'translate(-10,10) rotate(315)')
}

function remove_empty_bins(source_group) {
    return {
        all: function () {
            return source_group.all().filter(function (d) {
                return d.value != 0;
            });
        }
    };
}

var ingr_freq_barchart2 = function (data, bucket_size) {

    data.forEach(function (d) {
        var bucket = Math.ceil(+d.freq / bucket_size);
        d.bucket = ((bucket-1)*bucket_size).toString() + '-' + (bucket*bucket_size).toString()
    });

    ndx = crossfilter(data);
    comp_dim = ndx.dimension(function (d) { return d.comp; });
    var freq_dim = ndx.dimension(function (d) { return +d.freq; }),
        bucket_dim = ndx.dimension(function (d) { return d.bucket; }),
        nrcomp_bucket = bucket_dim.group().reduceCount(),
        nrcomp_comp = comp_dim.group().reduceSum(function (d) { return +d.freq; }),
        bucketnrcomp_comp = remove_empty_bins(nrcomp_comp);
    var minFreq = freq_dim.bottom(1)[0].freq;
    var maxFreq = freq_dim.top(1)[0].freq;

    var bucket_comp = ['00011484', '00012461'];

    ingr_bucket_piechart
        .dimension(bucket_dim)
        .group(nrcomp_bucket)
        .innerRadius(50);
//        .on("filtered", on_filteringr_bucket_piechart);

    var width = document.getElementById('box_ingr-freq-barchart2').offsetWidth;
    ingr_freq_barchart
        .dimension(comp_dim)
//        .group(nrcomp_comp)
        .group(bucketnrcomp_comp)
        .brushOn(false)
        .width(width)
        .margins({ top: 10, right: 10, bottom: 50, left: 50 })
        .x(d3.scale.ordinal())
//      .x(d3.scale.ordinal().domain(bucket_comp))
//        .xUnits(function(start, end, xDomain){return 10;})
//        .x(d3.scale.ordinal())
        .xUnits(dc.units.ordinal)
        .xAxisLabel('Components')
        .elasticX(true)
//        .y(d3.scale.linear().domain([minFreq, maxFreq]).tickFormat(d3.format("d")))
        .elasticY(true)
        .yAxisLabel('Frequency');


    ingr_freq_barchart.on("renderlet", on_renderletingr_freq_barchart);


    ingr_freq_table
        .dimension(comp_dim)
        .group(function (d) { return d.comp })
        .showGroups(false)
        .size(10)
        .columns([
          function (d) { return d.comp; },
          function (d) { return d.freq; },
          function (d) { return d.bucket; }
        ])
        .sortBy(function (d) { return d.freq })
        .order(d3.descending);

    dc.renderAll();
};

function ExploreDynamic() {
    d3.json('/api/e2equery_compfreq', function (error, e2e) {
        if (error) {
            console.log(error)
        }
//        d3.select('h2#data1-title').text('JSON Component Frequency');
//        d3.select('div#data1 pre').html(JSON.stringify(e2e, null, 4));
//        chartBars('#chart2', bardata)
//        ingr_freq_barchart1('#ingr-freq-barchart1', e2e)
        ingr_freq_barchart2(e2e, 100)
    });
}


d3.select('#version').text('v' + dc.version);
ExploreDynamic()