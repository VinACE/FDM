// explore.js

'use strict';

function getArgs() {
    var args = new Object();
    var query = location.search.substring(1);
    var pairs = query.split(",");
    for (var i = 0; i < pairs.length; i++) {
        var pos = pairs[i].indexOf('=');
        if (pos == -1) continue; // not found, skip
        var argname = pairs[i].substring(0, pos);
        var value = pairs[i].substring(pos+1);
        args[argname] = unescape(value);
    }
    return args;
}

var diameter = 960;

var tree = d3.layout.tree()
    .size([360, diameter / 2 - 120])
    .separation(function (a, b) { return (a.parent == b.parent ? 1 : 2) / a.depth; });

var diagonal = d3.svg.diagonal.radial()
    .projection(function (d) { return [d.y, d.x / 180 * Math.PI]; });

var args = getArgs();
var basket_id = args.basket;
var path = location.pathname;
if ( path[path.length - 1] == '/' ) path = path.slice(0, -1);
var basket = path.substr(path.lastIndexOf('/') + 1);

var svg = d3.select("#radial_tree").append("svg")
    .attr("width", diameter)
    .attr("height", diameter)
    .append("g")
    .attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")");

d3.json("/api/basket_experiment_api/"+basket, function (error, root) {
    if (error) {
        console.log(error)
    }

    var nodes = tree.nodes(root),
        links = tree.links(nodes);

    var link = svg.selectAll(".link")
        .data(links)
        .enter().append("path")
        .attr("class", "link")
        .attr("d", diagonal);

    var node = svg.selectAll(".node")
        .data(nodes)
        .enter().append("g")
        .attr("class", "node")
        .attr("transform", function (d) { return "rotate(" + (d.x - 90) + ")translate(" + d.y + ")"; })

    node.append("circle")
        .attr("r", 4.5);

    node.append("text")
        .attr("dy", ".31em")
        .attr("text-anchor", function (d) { return d.x < 180 ? "start" : "end"; })
        .attr("transform", function (d) { return d.x < 180 ? "translate(8)" : "rotate(180)translate(-8)"; })
        .text(function (d) { return d.name; });
});

d3.select(self.frameElement).style("height", diameter - 150 + "px");







