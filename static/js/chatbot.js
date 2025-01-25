document.getElementById('send-button').addEventListener('click', async () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input').value;

    // Display user's message
    chatMessages.innerHTML += `<div class="user-message">${userInput}</div>`;
    document.getElementById('user-input').value = '';

    try {
        const response = await fetch('/chatbot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_input: userInput, emotion: "Happy" }) // Replace with detected emotion if available
        });

        if (!response.ok) throw new Error(`HTTP error: ${response.statusText}`);

        const data = await response.json();
        chatMessages.innerHTML += `<div class="bot-message">${data.response}</div>`;
    } catch (error) {
        console.error("Chatbot error:", error);
        chatMessages.innerHTML += `<div class="bot-message error">Something went wrong. Please try again.</div>`;
    }
});
