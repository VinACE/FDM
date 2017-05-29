// script.js

'use strict';

// JQuery

$("#gen_retrieve_div").find("table").attr("id", "gen_retrieve_table")

$(document).ready(function () {

    $("#retrieve_results_div").find("table").attr("id", "retrieve_results_table").attr("class", "tablesorter-blue")

    var $table = $("#retrieve_results_table").tablesorter({
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




