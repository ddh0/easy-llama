let isGenerating = false;

function setIsGeneratingState(state) {
    isGenerating = state;
    const sendButton = document.querySelector('button[type="submit"]');
    const resetButton = document.getElementById('resetButton')

    if (state) {
        sendButton.textContent = 'cancel';
        sendButton.classList.add('cancel-button');
        resetButton.disabled = true
        resetButton.setAttribute('background-color', '#666666')
        updatePlaceholderText();
    } else {
        sendButton.textContent = 'send message';
        sendButton.classList.remove('cancel-button');
        resetButton.disabled = false
        resetButton.setAttribute('background-color', '#4b724b')
        updatePlaceholderText();
    }
}

function submitForm(event) { // this is called when `send message` OR `cancel` is clicked
    event.preventDefault(); // Prevent the default form submission
    const form = event.target;
    const formData = new FormData(form);
    const prompt = formData.get('prompt');

    if (isGenerating) {
        cancelGeneration();       // if already generating, cancel was clicked
        updatePlaceholderText();
        return
    } else {
        if (prompt == '') {
            return;           // do not submit empty prompts
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

// Wrap the reset button event listener in a function
function setupResetButton() {
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
                console.error('Not OK: setupResetButton:', response.statusText);
            }
        })
        .catch(error => {
            console.error('Caught error: setupResetButton:', error);
        });
    });
}

function updatePlaceholderText() {
    fetch('/get_placeholder_text')
        .then(response => response.json())
        .then(data => {
            const promptInput = document.getElementById('prompt');
            promptInput.placeholder = data.placeholder_text;
        })
        .catch(error => {
            console.error('Error fetching placeholder text:', error);
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
            if (response.ok ) {
                setIsGeneratingState(false);
                return;
            } else {
                console.error('Not OK: cancelGeneration:', response.statusText);
            }
        })
}

window.onload = setupResetButton;
