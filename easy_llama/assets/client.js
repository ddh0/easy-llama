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
const maxLengthInput = 1000000; // one million characters


function bytesToBase64(bytes) {
    let binaryString = '';
    for (let i = 0; i < bytes.length; i++) {
        binaryString += String.fromCharCode(bytes[i]);
    }
    return btoa(binaryString);
}


function base64ToBytes(base64) {
    //console.log('attempting atob', base64);
    const binString = atob(base64);
    const bytes = new Uint8Array(binString.length);
    for (let i = 0; i < binString.length; i++) {
        bytes[i] = binString.charCodeAt(i);
    }
    return bytes;
}


function fixBase64Padding(base64) {
    // remove all '=' characters
    let _base64_no_equals = base64.replace(/=/g, '');
    const missingPadding = _base64_no_equals.length % 4;
    if (missingPadding) {
        // add padding to the end if necessary
        _base64_no_equals += '='.repeat(4 - missingPadding);
    }
    return _base64_no_equals;
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
const summarizeButton = document.getElementById('summarizeButton');
const topKInput = document.getElementById('top_k');
const topPInput = document.getElementById('top_p');
const minPInput = document.getElementById('min_p');
const tempInput = document.getElementById('temp');


function popAlertPleaseReport(text) {
    alert(
        text + "\n\nPlease report this issue to the developer at this link:\n" +
        "https://github.com/ddh0/easy-llama/issues/new/choose"
    );
}


function handleError(message) { // TODO: make this smarter?
    if (!strHasContent(message)) {
        console.error('An error occured, but no error message was provided');
        return;
    } else {
        console.error('An error occured:', message);
        return;
    }
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
        summarizeButton.classList.add('disabled-button');
        summarizeButton.disabled = true;
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
        summarizeButton.classList.remove('disabled-button');
        summarizeButton.disabled = false;
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

    popAlertPleaseReport('cannot create message that is not from user or bot');

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
                handleError(
                    'Bad response from /remove: ' +
                    response.status +
                    response.statusText
                );
                reject();
            }
        })
        .catch(error => {
            handleError("removeLastMessage: " + error.message);
            reject();
        });
    });
}


function strHasContent(str) {
    if (!str || str === '') {
        return false;
    } else {
        return true;
    }
}


function highlightMessage(message) {
    // given a message, submit its `<pre><code>...</pre></code>` elements
    // to highlight.js
    const codeElements = message.querySelectorAll('pre code');
    codeElements.forEach(codeElement => {
        hljs.highlightElement(codeElement)
    });
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


function streamToMessage(reader, targetMessage, prefix) {
    let accumulatedText = '';
    let accumulatedBase64 = '';

    if (strHasContent(prefix)) {
        accumulatedText = prefix;
    }

    return new Promise((resolve, reject) => {
        function processStream({ done, value }) {
            if (done) {
                // Decode any remaining base64 data
                if (accumulatedBase64.length > 0) {
                    try {
                        accumulatedText += decode(
                            fixBase64Padding(accumulatedBase64)
                        );
                    } catch (error) {
                        handleError(
                            "Cannot decode remainder of base64: " +
                            accumulatedBase64
                        );
                    }
                }
                resolve();
                return;
            }

            let chunk = GlobalDecoder.decode(value);

            // Concatenate the base64 chunks
            accumulatedBase64 += chunk;

            // Split the accumulated base64 string by newlines
            let chunks = accumulatedBase64.split('\n');
            accumulatedBase64 = chunks.pop();  // Keep the last incomplete chunk

            for (const base64Chunk of chunks) {
                if (base64Chunk) {
                    try {
                        decoded_chunk = decode(fixBase64Padding(base64Chunk));
                        //console.log(
                        //    "single base64chunk",
                        //    base64Chunk,
                        //    "which is",
                        //    decoded_chunk
                        //);
                        accumulatedText += decoded_chunk
                    } catch (error) {
                        handleError(
                            "Decoding base64 chunk: " + error.name + " -- " +
                            error.message + " : " + base64Chunk
                        );
                    }
                }
            }

            //console.log('accumulated:', accumulatedText);

            targetMessage.innerHTML = marked.parse(accumulatedText);
            highlightMessage(targetMessage);

            // read the next chunk recursively
            reader.read().then(processStream);//.catch(error => {
               // handleError("streamToMessage run: " + error.message);
                //reject();
            //});
        }

        // start reading the stream
        reader.read().then(processStream);//.catch(error => {
        //    handleError('streamToMessage start: ' + error.message);
        //    reject();
        //});
    });
}


function submitForm(event) {
    event.preventDefault();
    const prompt = inputBox.value;

    if (isGenerating) {

        // if already generating, cancel was clicked
        cancelGeneration();
        return;

    } else {

        if (!strHasContent(prompt)) {
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

    // append user message to the conversation
    let userMessage = createMessage('user', prompt)
    highlightMessage(userMessage);
    appendNewMessage(userMessage);

    inputBox.value = '';

    // create a new bot message element
    const botMessage = createMessage('bot', '')
    appendNewMessage(botMessage);

    fetch('/submit', {
        method: 'POST',
        body: encode(prompt)
    })
    .then(response => {
        updatePlaceholderText();
        return streamToMessage(response.body.getReader(), botMessage, null);
    })
    .then(() => {
        setIsGeneratingState(false);
        updatePlaceholderText();
    })
    .catch(error => {
        handleError('submitForm: ' + error.message);
        setIsGeneratingState(false);
        updatePlaceholderText();
    });

}


function updatePlaceholderText() {

    if (isGenerating) {
        console.log('refuse to fetch context string - currently generating');
        return;
    }

    return fetch('/get_context_string')
        .then(response => response.json())
        .then(data => {
            inputBox.placeholder = decode(data.text);
        })
        .catch(error => {
            handleError('updatePlaceholderText: ' + error.message);
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
                handleError(
                    "cancelGeneration: bad response from /cancel: " +
                    response.status +
                    response.statusText
                );
                reject();
            }
        })
        .catch(error => {
            handleError('cancelGeneration: ' + error.message);
            reject();
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
        let botMessage = null;

        if (strHasContent(v)) { // trigger with bot prefix
            if (v.length > maxLengthInput) {
                alert(
                    'length of input exceeds maximum allowed length of ' +
                    '100k characters'
                );
                resolve();
                return;
            }

            encodedPrefix = encode(v);
            accumulatedText = v;

            botMessage = createMessage('bot', v);
            highlightMessage(botMessage);
            appendNewMessage(botMessage);
            inputBox.value = '';
        }

        setIsGeneratingState(true);

        fetch('/trigger', { method: 'POST', body: encodedPrefix })
        .then(response => {
            if (response.ok) {
                if (botMessage === null) {
                    botMessage = createMessage('bot', '');
                    appendNewMessage(botMessage);
                }

                // clear input box
                inputBox.value = '';

                return streamToMessage(
                    response.body.getReader(), botMessage, v
                );
            } else {
                handleError(
                    "Bad response from /trigger: " +
                    response.status +
                    response.statusText
                );
                reject();
            }
        })
        .then(() => {
            setIsGeneratingState(false);
            updatePlaceholderText();
            resolve();
        })
        .catch(error => {
            handleError("newBotMessage: " + error.message);
            setIsGeneratingState(false);
            updatePlaceholderText();
            reject();
        });
    });
}


function resetConversation() {
    fetch('/reset', { method : 'POST' })
    .then(response => {
        if (response.ok) {
            populateConversation();
        } else {
            handleError(
                "Bad response from /reset: " +
                response.status +
                response.statusText
            );
        }
    })
    .catch(error => {
        handleError("resetConversation: " + error.message);
    });
}


function populateConversation() {
    fetch('/convo', { method : "GET" })
    .then(response => {
        if (!response.ok) {
            handleError(
                "Bad response from /convo: " +
                response.status + 
                response.statusText
            );
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

            let role = decode(keys[0]);
            let content = decode(msg[keys[0]]);

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
        handleError("populateConversation: " + error.message);
        return;
    });
}


async function swipe() {

    if (isGenerating) {

        await cancelGeneration();
        setTimeout(newBotMessage, 750);
        await updatePlaceholderText();

    } else {

        await removeLastMessage();
        setTimeout(newBotMessage, 750);
        await updatePlaceholderText();
    }

}


function generateSummary() {
    return new Promise((resolve, reject) => {

        if (isGenerating) {
            console.log('refuse to generate summary - already generating');
            resolve();
            return;
        }

        setIsGeneratingState(true);

        fetch('/summarize', { method : 'GET' })
        .then(response => {
            if (response.ok) {
                return response.text(); // Read the response body as text
            } else {
                setIsGeneratingState(false);
                handleError(
                    "Bad response from /summarize: " +
                    response.status +
                    response.statusText
                );
                reject();
            }
        })
        .then(data => {
            let summary = decode(data); // Use the text data
            console.log('summary:', summary);
            alert(summary);
            setIsGeneratingState(false);
            resolve();
        })
        .catch(error => {
            handleError("generateSummary: " + error.message);
            setIsGeneratingState(false);
            reject();
        });
    });
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

    });

    submitButton.addEventListener('click', function(event) {
        submitForm(event);
    });

    removeButton.addEventListener('click', removeLastMessage);

    resetButton.addEventListener('click', function() {
        if (isGenerating) {
            cancelGeneration();
            resetConversation();
        } else {
            resetConversation();
        }
    });

    newBotMessageButton.addEventListener('click', newBotMessage);

    swipeButton.addEventListener('click', swipe);

    uploadButton.addEventListener('click', function() {
        uploadForm.click();
    });

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
                handleError("uploadForm: event change: " + error.message);
            }
        }
    });

    summarizeButton.addEventListener('click', generateSummary);

});


window.onresize = function() { 
    document.body.height = window.innerHeight;
}
