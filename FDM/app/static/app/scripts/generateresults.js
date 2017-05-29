// script.js

'use strict';

// JQuery

$("#gen_predict_div").find("table").attr("id", "gen_predict_table")

$(document).ready(function () {

    var $table = $("#gen_results_table").tablesorter({
        widgets: ["zebra", "filter", "resizable"],
        widgetOptions: {
            // class name applied to filter row and each input
            filter_cssFilter: 'tablesorter-filter',
            // search from beginning
            filter_startsWith: true,
            // Set this option to false to make the searches case sensitive
            filter_ignoreCase: true,
            filter_reset: '.reset',
            resizable_addLastColumn: true
        },
    });

    $("#gen_predict_div").find("table").attr("id", "gen_predict_table").attr("class", "tablesorter-blue")

    var $table = $("#gen_predict_table").tablesorter({
        widgets: ["zebra", "filter", "resizable"],
        widgetOptions: {
            // class name applied to filter row and each input
            filter_cssFilter: 'tablesorter-filter',
            // search from beginning
            filter_startsWith: true,
            // Set this option to false to make the searches case sensitive
            filter_ignoreCase: true,
            filter_reset: '.reset',
            resizable_addLastColumn: true
        },
    });

});




