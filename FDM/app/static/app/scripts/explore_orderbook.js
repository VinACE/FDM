// explore.js

'use strict';

var period_barchart = dc.barChart('#period_barchart');
var region_barchart = dc.barChart('#region_barchart');
//var nrcomp_region_piechart = dc.pieChart('#nrcomp_region_piechart');
//var nrcomp_custcorp_barchart = dc.barChart('#nrcomp_custcorp_barchart');
var ndx = crossfilter();


var dashboard_orderbook = function (data) {

    ndx = crossfilter(data);

    var period_dim = ndx.dimension(function (d) { return d.period; });
    var region_dim = ndx.dimension(function (d) { return d.region; });
    var bud_period = period_dim.group().reduceSum(function (d) { return +d.BUD; });
    var lyr_period = period_dim.group().reduceSum(function (d) { return +d.LYR; });
    var tyr_period = period_dim.group().reduceSum(function (d) { return +d.TYR; });
    var oih_period = period_dim.group().reduceSum(function (d) { return +d.OIH; });
    var ord_period = period_dim.group().reduceSum(function (d) { return +d.ORD; });
    var bud_region = region_dim.group().reduceSum(function (d) { return +d.BUD; });
    var lyr_region = region_dim.group().reduceSum(function (d) { return +d.LYR; });
    var tyr_region = region_dim.group().reduceSum(function (d) { return +d.TYR; });
    var oih_region = region_dim.group().reduceSum(function (d) { return +d.OIH; });
    var ord_region = region_dim.group().reduceSum(function (d) { return +d.ORD; });

    period_barchart
        .dimension(period_dim)
        .group(bud_period, "BUD")
        .margins({ top: 10, right: 10, bottom: 50, left: 50 })
        .x(d3.scale.ordinal())
        .xUnits(dc.units.ordinal)
        .xAxisLabel('Fiscal Period')
        .elasticX(true)
//        .y(d3.scale.linear().domain([minFreq, maxFreq]).tickFormat(d3.format("d")))
        .elasticY(true)
        .yAxisLabel('Budget')
        .legend(dc.legend().x(60).y(10).itemHeight(13).gap(5));

    region_barchart
        .dimension(region_dim)
        .group(bud_region, "BUD")
        .margins({ top: 10, right: 10, bottom: 50, left: 50 })
        .x(d3.scale.ordinal())
        .xUnits(dc.units.ordinal)
        .xAxisLabel('Fiscal Period')
        .elasticX(true)
//        .y(d3.scale.linear().domain([minFreq, maxFreq]).tickFormat(d3.format("d")))
        .elasticY(true)
        .yAxisLabel('Budget')
        .legend(dc.legend().x(60).y(10).itemHeight(13).gap(5));

    dc.renderAll();
};

d3.select('#version').text('v' + dc.version);
//d3.csv("/data/explore_components.csv", function (data) {
//    data.foreach (function (d) {component.push(d)})
//});

d3.json('/api/e2equery_orderbook', function (error, e2e) {
    if (error) {
        console.log(error)
    }
    dashboard_orderbook(e2e)
});




