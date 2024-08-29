// script.js
// https://github.com/ddh0/easy-llama/

let isGenerating = false;

function setIsGeneratingState(targetState) {

    const sendButton = document.querySelector('button[type="submit"]');
    const resetButton = document.getElementById('resetButton');

    if (targetState) {
        sendButton.textContent = 'cancel';
        sendButton.classList.add('cancel-button');
        resetButton.disabled = true;
        updatePlaceholderText();
    } else {
        sendButton.textContent = 'send message';
        sendButton.classList.remove('cancel-button');
        resetButton.disabled = false;
        updatePlaceholderText();
    }

    isGenerating = targetState;
}

function submitForm(event) { // this is called when `send message` OR `cancel` is clicked
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const prompt = formData.get('prompt');

    if (isGenerating) {   // if already generating, cancel was clicked

        console.log('cancel button clicked');
        cancelGeneration();       

        // get most recent message bubble
        const messages = document.querySelectorAll('.message');
        const lastMessage = messages[messages.length - 1];
        lastMessage.remove(); // remove cancelled message bubble

        updatePlaceholderText();

        return
    } else {
        if (prompt == '') {
            console.log('will not submit empty prompt');
            return;           // TODO: generate new bot message
        }
    }

    // Append user message to the conversation
    const conversation = document.getElementById('conversation');
    const userMessage = document.createElement('div');
    userMessage.className = 'message user-message';
    userMessage.innerText = escapeHtml(prompt);
    conversation.appendChild(userMessage);

    // Clear prompt input text box
    form.reset();

    // Create a new bot message element
    const botMessage = document.createElement('div');
    botMessage.className = 'message bot-message';
    conversation.appendChild(botMessage);

    // Scroll to the bottom of the conversation
    conversation.scrollTop = conversation.scrollHeight;

    let accumulatedText = '';

    setIsGeneratingState(true);

    fetch('/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams(formData),
    })
    .then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');

        function readStream() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    setIsGeneratingState(false);
                    return;
                }
                const chunk = decoder.decode(value, { stream: true });
                accumulatedText += chunk;

                // Trim and normalize the text content without stripping newlines
                const normalizedText = accumulatedText.replace(/[ \t]+/g, ' ').trim();

                // Re-parse and render the normalized content
                const renderedContent = marked.parse(normalizedText);
                botMessage.innerHTML = renderedContent; // Update the innerHTML with the rendered content

                conversation.scrollTop = conversation.scrollHeight;
                readStream();
            }).catch(error => {
                console.error('Error reading stream:', error);
                setIsGeneratingState(false);
            });
        }

        readStream();
    })
    .catch(error => {
        console.error('Caught error: submitForm:', error);
        setIsGeneratingState(false);
    });
}

function updatePlaceholderText() {
    fetch('/get_stats')
        .then(response => response.json())
        .then(data => {
            const promptInput = document.getElementById('prompt');
            promptInput.placeholder = data.text;
        })
        .catch(error => {
            console.error('Error fetching context usage stats:', error);
        });
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&")
        .replace(/</g, "<")
        .replace(/>/g, ">")
        .replace(/"/g, "\"")
        .replace(/'/g, "'");
}

function cancelGeneration() {
    fetch('/cancel', {
        method: 'POST'
    })
    .then(response => {
            if (response.ok) {
                setIsGeneratingState(false);
                return;
            } else {
                console.error('Not OK: cancelGeneration:', response.statusText);
            }
        });
}

// this is called when the page is (re-)loaded
window.onload = function pageSetup() {

    console.log('do pageSetup()');

    document.getElementById('resetButton').addEventListener('click', function(event) {
        fetch('/reset', {
            method: 'POST',
        })
        .then(response => {
            if (response.ok) {
                // Clear the conversation div
                document.getElementById('conversation').innerHTML = '';
                updatePlaceholderText();
            } else {
                console.error('Bad response from /reset:', response.statusText);
            }
        })
        .catch(error => {
            console.error('setupResetButton:', error);
        });
    });

    document.getElementById("removeButton").addEventListener('click', function(event) {

        // get most recent message bubble
        const messages = document.querySelectorAll('.message');
        const lastMessage = messages[messages.length - 1];
    
        // trigger the `remove()` route on the server
        fetch('/remove', {
        method: 'POST',
        })
        .then(response => {
        if (response.ok) {
            lastMessage.remove(); // remove last message bubble
            updatePlaceholderText();
        } else {
            console.error('Bad response from /remove:', response.statusText);
        }
        })
        .catch(error => {
        console.error('removeButton:', error);
        });
    });

}

function setDocumentBodyHeight() { 
    document.body.height = window.innerHeight;
    console.log('set document body height');
}

window.onresize = setDocumentBodyHeight;
