// script.js
// https://github.com/ddh0/easy-llama/

let isGenerating = false;

function setIsGeneratingState(targetState) {

    const resetButton = document.getElementById('resetButton');
    const removeButton = document.getElementById('removeButton');
    const submitButton = document.getElementById('submitButton');
    const newBotMessageButton = document.getElementById('newBotMessageButton');
    const swipeButton = document.getElementById('newBotMessageButton');

    if (targetState) {
        submitButton.textContent = 'cancel generation';
        submitButton.classList.add('red-button');
        resetButton.classList.add('disabled-button');
        resetButton.disabled = true;
        removeButton.classList.add('disabled-button');
        removeButton.disabled = true;
        newBotMessageButton.classList.add('disabled-button');
        newBotMessageButton.disabled = true;
        swipeButton.classList.add('disabled-button')
        swipeButton.disabled = true;
        updatePlaceholderText();
    } else {
        submitButton.textContent = 'send message';
        submitButton.classList.remove('red-button');
        resetButton.classList.remove('disabled-button');
        resetButton.disabled = false;
        removeButton.classList.remove('disabled-button');
        removeButton.disabled = false;
        newBotMessageButton.classList.remove('disabled-button');
        newBotMessageButton.disabled = false;
        swipeButton.classList.remove('disabled-button')
        swipeButton.disabled = false;
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
            return;
        }
    }

    // Append user message to the conversation
    const conversation = document.getElementById('conversation');
    const userMessage = document.createElement('div');
    userMessage.className = 'message user-message';
    userMessage.innerHTML = marked.parse(prompt); // TODO: render markdown in user message bubble
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

                // we don't want to parse extra whitespace in markdown
                const normalizedText = accumulatedText.replace(/[ \t]+/g, ' ').trim();

                const renderedContent = marked.parse(normalizedText);
                botMessage.innerHTML = renderedContent;

                // conversation.scrollTop = conversation.scrollHeight;
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

function newBotMessage(event) {

    // do not trigger generation if already generating
    if (isGenerating) { return } else {

        // Create a new bot message element
        const botMessage = document.createElement('div');
        botMessage.className = 'message bot-message';
        conversation.appendChild(botMessage);

        // Scroll to the bottom of the conversation
        conversation.scrollTop = conversation.scrollHeight;

        let accumulatedText = '';

        setIsGeneratingState(true);

        fetch('/trigger', {
            method: 'POST',  
        })
        .then(response => {
            if (response.ok) {
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
        
                        // we don't want to parse extra whitespace in markdown
                        const normalizedText = accumulatedText.replace(/[ \t]+/g, ' ').trim();
        
                        const renderedContent = marked.parse(normalizedText);
                        botMessage.innerHTML = renderedContent;
        
                        // conversation.scrollTop = conversation.scrollHeight;
                        readStream();
                    }).catch(error => {
                        console.error('Error reading stream:', error);
                        setIsGeneratingState(false);
                    });
                }
        
                readStream();

            } else {
                console.error('Bad response from /trigger:', response.statusText);
            }
        });
    }
}

function removeLastMessage() {

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
}

function resetConversation() {
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
}

window.onload = function pageSetup() {

    console.log('do pageSetup()');

    document.getElementById('resetButton').addEventListener('click', resetConversation);

    document.getElementById("removeButton").addEventListener('click', removeLastMessage);

    document.getElementById("newBotMessageButton").addEventListener('click', newBotMessage);

    document.getElementById('prompt').addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            document.getElementById('promptForm').dispatchEvent(new Event('submit'));
        }
    });

    document.getElementById('swipeButton').addEventListener('click', function(event) {
        if (isGenerating) { return } else {
            removeLastMessage();
            newBotMessage();
        }
    });

}

window.onresize = function setDocumentBodyHeight() { 
    document.body.height = window.innerHeight;
    console.log('set document body height');
}
