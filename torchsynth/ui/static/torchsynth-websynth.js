
function get(action, callback) {
    $.ajax({
        url: "/" + action,
    }).done(function (data) {
        callback(data);
    });
}

function post(action, data, callback) {
    $.post("/" + action, data, callback);
}

function add_parameters(parameters) {
    let controls = $('#controls');
    controls.empty();
    let modules = {};

    for (const [key, value] of Object.entries(parameters)) {
        if (!(value.name[0] in modules)) {
            modules[value.name[0]] = {};
        }
        modules[value.name[0]][value.name[1]] = value;
    }

    for (const [module_name, module] of Object.entries(modules)) {
        let newModule = $('<div class="demo-card-wide mdl-card mdl-shadow--2dp" style="padding: 25px;"></div>');
        newModule.append(`<b style="padding-bottom: 10px;">${module_name}</b>`);

        for (const [key, value] of Object.entries(module)) {
            newModule.append(`<div class="mdl-card__supporting-text">${value.name[1]}</div>`)
            let newControl = $('<p style="width:300px"></p>');
            newControl.append($(`<input class="mdl-slider mdl-js-slider torchsynth-param" type="range" id="${value.name[0]}-${value.name[1]}" min="${value.min}" max="${value.max}" value="${value.val}" step="0.001">`));
            newModule.append(newControl);
        }
        controls.append(newModule);
    }
    componentHandler.upgradeDom();
}


$(document).ready(() => {
    get("parameters", (data) => {
        console.log(data);
        add_parameters(data);
    });

    $('#render-patch').click(() => {
        patch = {};
        $('.torchsynth-param').each(function(index) {
            patch[$(this).attr("id")] = $(this).val();
        });
        $('#audio-player').empty();
        post('set_patch', patch, (data) => {
            console.log("here");
            $('#audio-player').append('<audio controls autoplay><source src="get_rendered" type="audio/wav"></audio>');
        });
    });

    document.addEventListener('keydown', function(event) {
        if (event.code == 'KeyF') {
            $('#render-patch').click();
        }
    });


});