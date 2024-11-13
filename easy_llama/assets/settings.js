// settings.js
// https://github.com/ddh0/easy-llama/

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('samplerForm');

    fetch('/get_sampler')
        .then(response => response.json())
        .then(data => {
            document.getElementById('max_len_tokens').value = data.max_len_tokens;
            document.getElementById('top_k').value = data.top_k;
            document.getElementById('top_p').value = data.top_p;
            document.getElementById('min_p').value = data.min_p;
            document.getElementById('temp').value = data.temp;
            document.getElementById('frequency_penalty').value = data.frequency_penalty;
            document.getElementById('presence_penalty').value = data.presence_penalty;
            document.getElementById('repeat_penalty').value = data.repeat_penalty;
        })
        .catch(error => console.error('error fetching sampler settings:', error));

    form.addEventListener('submit', function(event) {
        event.preventDefault();

        const data = {
            max_len_tokens: parseInt(document.getElementById('max_len_tokens').value),
            top_k: parseInt(document.getElementById('top_k').value),
            top_p: parseFloat(document.getElementById('top_p').value),
            min_p: parseFloat(document.getElementById('min_p').value),
            temp: parseFloat(document.getElementById('temp').value),
            frequency_penalty: parseFloat(document.getElementById('frequency_penalty').value),
            presence_penalty: parseFloat(document.getElementById('presence_penalty').value),
            repeat_penalty: parseFloat(document.getElementById('repeat_penalty').value)
        };

        fetch('/update_sampler', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            if (response.ok) {
                alert('sampler settings updated successfully');
            } else {
                alert('failed to update sampler settings');
            }
        })
        .catch(error => console.error('error updating sampler settings:', error));
    });

    document.getElementById('setDefaultsButton').addEventListener('click', function() {
        document.getElementById('max_len_tokens').value = -1;
        document.getElementById('top_k').value = -1;
        document.getElementById('top_p').value = 0.95;
        document.getElementById('min_p').value = 0.05;
        document.getElementById('temp').value = 0.8;
        document.getElementById('frequency_penalty').value = 0.0;
        document.getElementById('presence_penalty').value = 0.0;
        document.getElementById('repeat_penalty').value = 1.0;
    });

    document.getElementById('neutralizeAllButton').addEventListener('click', function() {
        document.getElementById('max_len_tokens').value = -1;
        document.getElementById('top_k').value = -1;
        document.getElementById('top_p').value = 1.0;
        document.getElementById('min_p').value = 0.0;
        document.getElementById('temp').value = 1.0;
        document.getElementById('frequency_penalty').value = 0.0;
        document.getElementById('presence_penalty').value = 0.0;
        document.getElementById('repeat_penalty').value = 1.0;
    });
});