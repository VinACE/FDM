// explore.js

'use strict';

var nrcomp_fiscper_barchart = dc.barChart('#nrcomp_fiscper_barchart');
var nrcomp_catclass_piechart = dc.pieChart('#nrcomp_catclass_piechart');
var nrcomp_region_piechart = dc.pieChart('#nrcomp_region_piechart');
var nrcomp_custcorp_barchart = dc.barChart('#nrcomp_custcorp_barchart');
var ndx = crossfilter();
var comp_dim;
var components;


function on_renderlet_nrcomp_fiscper_barchart() {
    // rotate x-axis labels
    nrcomp_fiscper_barchart.selectAll('g.x text')
        .attr('transform', 'translate(-10,10) rotate(315)')
}

function on_renderlet_nrcomp_custcorp_barchart() {
    // rotate x-axis labels
    nrcomp_custcorp_barchart.selectAll('g.x text')
        .attr('transform', 'translate(-10,10) rotate(315)')
}

function fakedim_top_x(source_group, x) {
    return {
        all: function () {
            return source_group.top(x);
        }
    };
}

var dashboard_e2esubm = function (data) {

    ndx = crossfilter(data);
    comp_dim = ndx.dimension(function (d) { return d.component; });
    var catclass_dim = ndx.dimension(function (d) { return d.catclass; });
    var fiscper_dim = ndx.dimension(function (d) { return d.fiscper; });
    var region_dim = ndx.dimension(function (d) { return d.region; });
    var custcorp_dim = ndx.dimension(function (d) { return d.custcorp; });
    var nrcomp_catclass = catclass_dim.group().reduceSum(function (d) { return +d.count; });
    var nrcomp_fiscper = fiscper_dim.group().reduceSum(function (d) { return +d.count; });
    var nrcomp_region = region_dim.group().reduceSum(function (d) { return +d.count; });
    var nrcomp_custcorp = custcorp_dim.group().reduceSum(function (d) { return +d.count; });
    var top_nrcomp_custcorp = fakedim_top_x(nrcomp_custcorp, 20)
    var nrcomp_fiscper_001 = fiscper_dim.group().reduceSum(function (d) {
        var nrcomp =  d.catclass == '001' ? d.count : 0;
        return nrcomp; });
    var nrcomp_fiscper_002 = fiscper_dim.group().reduceSum(function (d) {
        var nrcomp = d.catclass == '002' ? d.count : 0;
        return nrcomp;
    });
    var nrcomp_fiscper_003 = fiscper_dim.group().reduceSum(function (d) {
        var nrcomp = d.catclass == '003' ? d.count : 0;
        return nrcomp;
    })

    nrcomp_fiscper_barchart
        .dimension(fiscper_dim)
        .group(nrcomp_fiscper_001, "001")
        .stack(nrcomp_fiscper_002, "002")
        .stack(nrcomp_fiscper_003, "003")
        .margins({ top: 10, right: 10, bottom: 50, left: 50 })
        .x(d3.scale.ordinal())
        .xUnits(dc.units.ordinal)
        .xAxisLabel('Fiscal Period')
        .elasticX(true)
//        .y(d3.scale.linear().domain([minFreq, maxFreq]).tickFormat(d3.format("d")))
        .elasticY(true)
        .yAxisLabel('Nr Components')
        .legend(dc.legend().x(60).y(10).itemHeight(13).gap(5))
        .on("renderlet", on_renderlet_nrcomp_fiscper_barchart);


    nrcomp_catclass_piechart
        .dimension(catclass_dim)
        .group(nrcomp_catclass)
        .innerRadius(10);

    nrcomp_region_piechart
    .dimension(region_dim)
    .group(nrcomp_region)
    .innerRadius(10);

    var max_nrcomp_custcorp = nrcomp_custcorp.top(1)[0].count;
    var width = document.getElementById('container').offsetWidth;
    nrcomp_custcorp_barchart
        .dimension(custcorp_dim)
        .group(top_nrcomp_custcorp)
        .width(width)
        .margins({ top: 10, right: 10, bottom: 50, left: 50 })
        .x(d3.scale.ordinal())
        .xUnits(dc.units.ordinal)
        .xAxisLabel('Cust Corporation')
        .elasticX(true)
        .brushOn(true)
        .elasticY(true)
        .yAxisLabel('Nr Components')
        .on("renderlet", on_renderlet_nrcomp_custcorp_barchart);



    dc.renderAll();
};

d3.select('#version').text('v' + dc.version);
//d3.csv("/data/explore_components.csv", function (data) {
//    data.foreach (function (d) {component.push(d)})
//});

d3.json('/api/e2equery_e2esubm', function (error, e2e) {
    if (error) {
        console.log(error)
    }
    dashboard_e2esubm(e2e)
});




