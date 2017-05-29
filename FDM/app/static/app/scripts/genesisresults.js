// script.js

'use strict';

// JQuery


$('[name^="select_all"]').click(function () {
    $('[name ^= "selected[]"]').prop('checked', true);
})

$('[name^="select_none"]').click(function () {
    $('[name ^= "selected[]"]').prop('checked', false);
})

$(document).ready(function () {

    var $table = $("#genesis_comp_table").tablesorter({
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

    var $table = $("#genesis_mat_table").tablesorter({
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




