// script.js

'use strict';

// JQuery

$(document).ready(function () {

    var $table = $("#gen_experiment_table").tablesorter({
        headers: {
            0: { sorter: true },
            1: { sorter: true },
            2: { sorter: true }
        },
        // sort on the bomparts, descending 
        sortList: [[2,1]],
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




