// script.js

'use strict';

// JQuery


function display_message(data) {
    $("#basket_progress").append("<p>" + data + "</p>");
}

function poll() {
    var poll_interval = 0;

    $.ajax({
        url: "api/basket_pollresults",
        type: 'GET',
        dataType: 'json',
        cache: false,
        success: function (data) {
            display_message(data);
            poll_interval = 0;
        },
        error: function () {
            poll_interval = 1000;
        },
        complete: function () {
            setTimeout(poll, poll_interval);
        },
    });
}


$('button[name^="generate"]').click(function () {
    poll();
})

$(document).ready(function () {
//    poll();
});




