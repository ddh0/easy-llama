// client.js
// https://github.com/ddh0/easy-llama/

marked.setOptions({
    pedantic: false,  // more relaxed parsing
    gfm: true,        // github-flavored markdown
    breaks: true      // insert <br> on '\n' also, not only on '\n\n'
});


document.body.height = window.innerHeight;


let isGenerating = false;

const GlobalEncoder  = new TextEncoder;
const GlobalDecoder  = new TextDecoder;
const maxLengthInput = 100000; // one hundred thousand characters


function bytesToBase64(bytes) {
    let binaryString = '';
    for (let i = 0; i < bytes.length; i++) {
        binaryString += String.fromCharCode(bytes[i]);
    }
    return btoa(binaryString);
}


function base64ToBytes(base64) {
    const binString = atob(base64);
    const bytes = new Uint8Array(binString.length);
    for (let i = 0; i < binString.length; i++) {
        bytes[i] = binString.charCodeAt(i);
    }
    return bytes;
}


function encode(text) {
    // utf-8 str -> utf-8 bytes -> base64 str
    let bytes = GlobalEncoder.encode(text);
    return bytesToBase64(bytes);
}


function decode(base64) {
    // base64 str -> utf-8 bytes -> utf-8 str
    let bytes = base64ToBytes(base64);
    return GlobalDecoder.decode(bytes);
}


const conversationWindow = document.getElementById('conversationWindow');
const resetButton = document.getElementById('resetButton');
const removeButton = document.getElementById('removeButton');
const submitButton = document.getElementById('submitButton');
const newBotMessageButton = document.getElementById('newBotMessageButton');
const swipeButton = document.getElementById('swipeButton');
const inputBox = document.getElementById('inputBox');
const inputForm = document.getElementById('inputForm');
const uploadButton = document.getElementById('fileUploadButton');
const uploadForm = document.getElementById('fileInput');


function popAlertPleaseReport(text) {
    alert(
        text + "\n\nPlease report this issue to the developer at this link:\n" +
        "https://github.com/ddh0/easy-llama/issues/new/choose"
    );
}


function setIsGeneratingState(targetState) {

    if (isGenerating === targetState) { return; }

    if (targetState) {
        submitButton.textContent = 'cancel generation';
        submitButton.classList.add('red-button');
        //resetButton.classList.add('disabled-button');
        //resetButton.disabled = true;
        resetButton.textContent = 'cancel and restart chat';
        removeButton.classList.add('disabled-button');
        removeButton.disabled = true;
        newBotMessageButton.classList.add('disabled-button');
        newBotMessageButton.disabled = true;
        swipeButton.textContent = 'cancel and re-roll';
        //swipeButton.classList.add('disabled-button');
        //swipeButton.disabled = true;
        //uploadButton.classList.add('disabled-button');
        //uploadButton.disabled = true;
    } else {
        submitButton.textContent = 'send message';
        submitButton.classList.remove('red-button');
        //resetButton.classList.remove('disabled-button');
        //resetButton.disabled = false;
        resetButton.textContent = 'restart chat';
        removeButton.classList.remove('disabled-button');
        removeButton.disabled = false;
        newBotMessageButton.classList.remove('disabled-button');
        newBotMessageButton.disabled = false;
        swipeButton.textContent = 're-roll last message';
        //swipeButton.classList.remove('disabled-button');
        //swipeButton.disabled = false;
        //uploadButton.classList.remove('disabled-button');
        //uploadButton.disabled = false;
    }

    isGenerating = targetState;
    console.log('set generating state:', targetState)
}


function getMostRecentMessage() { 
    return conversationWindow.firstChild; // node or null
}


function appendNewMessage(message) {
    let mostRecentMessage = getMostRecentMessage();
    if (mostRecentMessage !== null) {
        conversationWindow.insertBefore(message, mostRecentMessage);
    } else {
        conversationWindow.append(message);
    }
}


function createMessage(role, content) {

    if (role == 'user') {
        const message = document.createElement('div');
        message.className = 'message user-message';
        message.innerHTML = marked.parse(content);
        return message;
    }

    if (role == 'bot') {
        const message = document.createElement('div');
        message.className = 'message bot-message';
        message.innerHTML = marked.parse(content);
        return message;
    }

    popAlertPleaseReport('unreachable in createMessage');

}


function removeLastMessage() {
    return new Promise((resolve, reject) => {
        const lastMessage = getMostRecentMessage();

        if (lastMessage === null) {
            resolve();
            return;
        }

        fetch('/remove', {
            method: 'POST',
        })
        .then(response => {
            if (response.ok) {
                lastMessage.remove();
                console.log('removeLastMessage: removed node');
                updatePlaceholderText();
                resolve();
            } else {
                reject(
                    new Error(
                        'Bad response from /remove: ' + response.statusText
                    )
                );
            }
        })
        .catch(error => {
            reject(new Error('removeButton: ' + error));
        });
    });
}


function highlightMessage(message) {
    // given a message, submit its `<pre><code>...</pre></code>` elements
    // to highlight.js
    const codeElements = message.querySelectorAll('pre code');
    codeElements.forEach(codeElement => {
        hljs.highlightElement(codeElement)
    });
}


function triggerFileUpload() {
    uploadForm.click();
}


function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            resolve(e.target.result);
        };
        reader.onerror = function(e) {
            reject(e);
        };
        reader.readAsText(file);
    });
}


function submitForm(event) {
    event.preventDefault();
    const form = inputForm;
    const formData = new FormData(form);
    const prompt = formData.get('prompt');

    if (isGenerating) {

        // if already generating, cancel was clicked
        cancelGeneration();
        return;

    } else {

        if ('' == prompt) {
            // if user hits submit with no prompt, trigger new bot message
            newBotMessage();
            updatePlaceholderText();
            return;
        }
    }

    if (prompt.length > maxLengthInput) {
        alert(
            'length of input exceeds maximum allowed length of ' + 
            '100k characters'
        );
        return;
    }

    setIsGeneratingState(true);

    // Append user message to the conversation
    let userMessage = createMessage('user', prompt)
    highlightMessage(userMessage);
    appendNewMessage(userMessage);

    // Clear prompt input text box
    form.reset();

    // Create a new bot message element
    const botMessage = createMessage('bot', '')
    appendNewMessage(botMessage);

    // Scroll to the bottom of the conversation
    conversationWindow.scrollTop = conversationWindow.scrollHeight;

    let accumulatedText = '';

    fetch('/submit', {
        method: 'POST',
        body: bytesToBase64(GlobalEncoder.encode(prompt))
    })
    .then(response => {

        updatePlaceholderText();

        const reader = response.body.getReader();

        function readStream() {
        
            reader.read().then(({ done, value }) => {
                if (done) {
                    setIsGeneratingState(false);
                    updatePlaceholderText();
                    return;
                }
        
                accumulatedText += GlobalDecoder.decode(base64ToBytes(
                    GlobalDecoder.decode(
                        value, { stream: true }
                    )
                ));

                botMessage.innerHTML = marked.parse(accumulatedText);
                highlightMessage(botMessage);
        
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
    return fetch('/get_context_string')
        .then(response => response.json())
        .then(data => {
            inputBox.placeholder = decode(data.text);
        })
        .catch(error => {
            console.error('Error fetching context usage string:', error);
        });
}


function escapeHtml(unsafe) { // currently unused
    return unsafe
        .replace(/&/g, "&")
        .replace(/</g, "<")
        .replace(/>/g, ">")
        .replace(/"/g, "\"")
        .replace(/'/g, "'");
}


function cancelGeneration() {
    return new Promise((resolve, reject) => {
        if (!isGenerating) {
            resolve();
            return;
        }

        fetch('/cancel', {
            method: 'POST'
        })
        .then(response => {
            if (response.ok) {
                setIsGeneratingState(false);

                // remove canceled message
                const lastMessage = getMostRecentMessage();
                if (lastMessage === null) {
                    resolve();
                } else {
                    lastMessage.remove();
                    resolve();
                }
            } else {
                reject(new Error(
                    'error when canceling generation: ' + response.statusText
                ));
            }
        })
        .catch(error => {
            reject(new Error('Error in cancelGeneration: ' + error));
        });
    });
}



function newBotMessage() {
    return new Promise((resolve, reject) => {
        if (isGenerating) {
            console.log('refuse to trigger newBotMessage - already generating');
            resolve();
            return;
        }

        let v = inputBox.value;
        let encodedPrefix = null;
        let accumulatedText = '';
        let botMessage = null;

        if (inputBox.value !== '') { // trigger with bot prefix
            if (v.length > maxLengthInput) {
                alert(
                    'length of input exceeds maximum allowed length of ' +
                    '100k characters'
                );
                resolve();
                return;
            }

            encodedPrefix = bytesToBase64(GlobalEncoder.encode(v));
            accumulatedText = v;

            botMessage = createMessage('bot', v);
            highlightMessage(botMessage);
            appendNewMessage(botMessage);
        }

        setIsGeneratingState(true);

        fetch('/trigger', { method : 'POST', body : encodedPrefix })
        .then(response => {
            if (response.ok) {
                if (botMessage === null) {
                    botMessage = createMessage('bot', '');
                    appendNewMessage(botMessage);
                }

                // clear input box
                inputBox.value = '';

                // Scroll to the bottom of the conversation
                conversationWindow.scrollTop = conversationWindow.scrollHeight;

                const reader = response.body.getReader();

                function readStream() {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            setIsGeneratingState(false);
                            updatePlaceholderText();
                            resolve();
                            return;
                        }

                        accumulatedText += GlobalDecoder.decode(
                            base64ToBytes(
                                GlobalDecoder.decode(
                                    value, { stream: true }
                                )
                            )
                        );

                        botMessage.innerHTML = marked.parse(accumulatedText);
                        highlightMessage(botMessage);

                        readStream();

                    }).catch(error => {
                        console.error('Error reading stream:', error);
                        setIsGeneratingState(false);
                        reject(error);
                    });
                }

                readStream();
            
            } else {
                console.error(
                    'Bad response from /trigger:', response.statusText
                );
                setIsGeneratingState(false);
                reject(new Error(
                    'Bad response from /trigger: ' + response.statusText
                ));
            }
        }).catch(error => {
            setIsGeneratingState(false);
            reject(new Error('Error in newBotMessage: ' + error));
        });
    });
}


function resetConversation() {
    fetch('/reset', { method : 'POST' })
    .then(response => {
        if (response.ok) {
            conversationWindow.innerHTML = '';
            updatePlaceholderText();
        } else {
            console.error('Bad response from /reset:', response.statusText);
        }
    })
    .catch(error => {
        console.error('Error in resetConversation:', error);
    });
}


function populateConversation() {
    fetch('/convo', { method : "GET" })
    .then(response => {
        if (!response.ok) {
            console.error('Bad response from /convo:', response.statusText);
            return;
        } else {
            updatePlaceholderText();

            return response.json(); // Return the promise from response.json()
        }
    })
    .then(data => {
        let msgs = Object.keys(data);

        conversationWindow.innerHTML = '';

        // iterate over all messages and add them to the conversation
        for (let i = 0; i < msgs.length; i++) {
            const msgKey = msgs[i];
            const msg = data[msgKey];
            let keys = Object.keys(msg);

            let role = GlobalDecoder.decode(
                base64ToBytes(keys[0])
            );
            let content = GlobalDecoder.decode(
                base64ToBytes(msg[keys[0]])
            );

            //console.log('i:', i, 'role:', role, 'content:', content);

            if (role != 'system') {
                let newMessage = createMessage(role, content);
                highlightMessage(newMessage);
                appendNewMessage(newMessage);

            }
        }

        return;
    })
    .catch(error => {
        console.error('Error in populateConversation:', error);
        return;
    });
}


async function swipe() {

    if (isGenerating) {

        await cancelGeneration();
        setTimeout(newBotMessage, 500);
        await updatePlaceholderText();

    } else {

        await removeLastMessage();
        setTimeout(newBotMessage, 500);
        await updatePlaceholderText();
    }

}


document.addEventListener('DOMContentLoaded', function() {

    // display all non-system messages even after page load/reload
    populateConversation();

    // SHIFT + ENTER -> newline
    // ENTER         -> submit form
    inputBox.addEventListener('keydown', function(event) {

        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // default is newline
            inputForm.dispatchEvent(
                new Event('submit')
            );
        }

        //if (inputBox.value === '') {
        //    submitButton.classList.add('disabled-button')
        //    submitButton.disabled = true
        //} else {
        //    submitButton.classList.remove('disabled-button')
        //    submitButton.disabled = false
        //}

    });

    submitButton.addEventListener('click', function(event) {
        submitForm(event);
    });

    removeButton.addEventListener('click', removeLastMessage);

    resetButton.addEventListener('click', function(event) {
        if (isGenerating) {
            cancelGeneration();
            resetConversation();
        } else {
            resetConversation();
        }
    });

    newBotMessageButton.addEventListener('click', newBotMessage);

    swipeButton.addEventListener('click', swipe);

    uploadButton.addEventListener('click', triggerFileUpload);

    uploadForm.addEventListener('change', async function(event) {
        const files = event.target.files;
        if (files.length > 0) {
            try {
                for (const file of files) {
                    const content = await readFileAsText(file);
                    inputBox.value += "```\n" + content + "\n```\n\n";
                }
                inputBox.scrollTop = inputBox.scrollHeight;
                inputBox.selectionStart = inputBox.value.length;
                inputBox.selectionEnd = inputBox.value.length;
            } catch (error) {
                console.error('Error reading file:', error);
            }
        }
    });

});


window.onresize = function() { 
    document.body.height = window.innerHeight;
}
