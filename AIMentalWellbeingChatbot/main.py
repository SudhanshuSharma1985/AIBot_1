from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List
import uvicorn
from collections import defaultdict
from datetime import datetime

from auth import create_access_token, get_current_user
from chatbot import WellnessChatbot

app = FastAPI(title="Social & Emotional Wellness Chatbot")

# Initialize chatbot
chatbot = WellnessChatbot()

# In-memory storage for chat history (per session)
chat_sessions: Dict[str, List[dict]] = defaultdict(list)

# Pydantic models
class UserLogin(BaseModel):
    username: str

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float

class Token(BaseModel):
    access_token: str
    token_type: str

# Routes
@app.post("/token", response_model=Token)
async def login(user_data: UserLogin):
    # Simple username-based login without password
    if not user_data.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username cannot be empty"
        )
    
    access_token = create_access_token(data={"sub": user_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: ChatMessage,
    current_user: str = Depends(get_current_user)
):
    # Detect intent
    intent_data = chatbot.detect_intent(message.message)
    intent = intent_data["intent"]
    confidence = intent_data.get("confidence", 0.8)
    
    # Generate response
    response = chatbot.generate_response(message.message, intent)
    
    # Store in session memory
    chat_sessions[current_user].append({
        "message": message.message,
        "response": response,
        "intent": intent,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Keep only last 100 messages per user
    if len(chat_sessions[current_user]) > 100:
        chat_sessions[current_user] = chat_sessions[current_user][-100:]
    
    return ChatResponse(
        response=response,
        intent=intent,
        confidence=confidence
    )

@app.get("/chat-history")
async def get_chat_history(
    current_user: str = Depends(get_current_user),
    limit: int = 50
):
    history = chat_sessions.get(current_user, [])
    return history[-limit:]

@app.post("/clear-history")
async def clear_history(current_user: str = Depends(get_current_user)):
    if current_user in chat_sessions:
        chat_sessions[current_user] = []
    return {"message": "Chat history cleared"}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wellness Chatbot</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f0f2f5;
            }
            .container {
                background-color: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c5282;
                text-align: center;
            }
            .auth-form {
                max-width: 400px;
                margin: 0 auto;
            }
            input {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-sizing: border-box;
            }
            button {
                width: 100%;
                padding: 12px;
                background-color: #3182ce;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #2c5282;
            }
            .chat-container {
                display: none;
                max-width: 600px;
                margin: 0 auto;
            }
            .messages {
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
                background-color: #f8f9fa;
            }
            .message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
            }
            .user-message {
                background-color: #3182ce;
                color: white;
                text-align: right;
                margin-left: 20%;
            }
            .bot-message {
                background-color: #e2e8f0;
                margin-right: 20%;
            }
            .intent-badge {
                font-size: 12px;
                padding: 2px 8px;
                border-radius: 12px;
                display: inline-block;
                margin-top: 5px;
            }
            .emergency {
                background-color: #f56565;
                color: white;
            }
            .irrelevant {
                background-color: #ed8936;
                color: white;
            }
            .qna {
                background-color: #48bb78;
                color: white;
            }
            .chat-input {
                display: flex;
                gap: 10px;
            }
            .chat-input input {
                flex: 1;
                margin: 0;
            }
            .chat-input button {
                width: auto;
                padding: 10px 20px;
            }
            .chat-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            .username-display {
                color: #2c5282;
                font-weight: bold;
            }
            .action-buttons {
                display: flex;
                gap: 10px;
                margin-top: 20px;
            }
            .action-buttons button {
                width: auto;
                padding: 10px 20px;
            }
            .logout-btn {
                background-color: #e53e3e;
            }
            .clear-btn {
                background-color: #ed8936;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Social & Emotional Wellness Chatbot</h1>
            
            <div id="authContainer" class="auth-form">
                <h2>Enter Your Username</h2>
                <input type="text" id="username" placeholder="Username" onkeypress="if(event.key==='Enter') login()">
                <button onclick="login()">Start Chat</button>
            </div>
            
            <div id="chatContainer" class="chat-container">
                <div class="chat-header">
                    <span class="username-display">Welcome, <span id="usernameDisplay"></span>!</span>
                </div>
                <div class="messages" id="messages"></div>
                <div class="chat-input">
                    <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="if(event.key==='Enter') sendMessage()">
                    <button onclick="sendMessage()">Send</button>
                </div>
                <div class="action-buttons">
                    <button class="clear-btn" onclick="clearHistory()">Clear History</button>
                    <button class="logout-btn" onclick="logout()">Logout</button>
                </div>
            </div>
        </div>
        
        <script>
            let token = sessionStorage.getItem('token');
            let username = sessionStorage.getItem('username');
            
            if (token && username) {
                showChat();
                loadChatHistory();
            }
            
            async function login() {
                const usernameInput = document.getElementById('username').value.trim();
                
                if (!usernameInput) {
                    alert('Please enter a username');
                    return;
                }
                
                try {
                    const response = await fetch('/token', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ username: usernameInput })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        sessionStorage.setItem('token', data.access_token);
                        sessionStorage.setItem('username', usernameInput);
                        token = data.access_token;
                        username = usernameInput;
                        showChat();
                        loadChatHistory();
                    } else {
                        alert('Login failed');
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            function showChat() {
                document.getElementById('authContainer').style.display = 'none';
                document.getElementById('chatContainer').style.display = 'block';
                document.getElementById('usernameDisplay').textContent = username;
                document.getElementById('username').value = '';
            }
            
            function logout() {
                sessionStorage.removeItem('token');
                sessionStorage.removeItem('username');
                token = null;
                username = null;
                document.getElementById('authContainer').style.display = 'block';
                document.getElementById('chatContainer').style.display = 'none';
                document.getElementById('messages').innerHTML = '';
            }
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                addMessage(message, 'user');
                input.value = '';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${token}`,
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        addMessage(data.response, 'bot', data.intent);
                    } else if (response.status === 401) {
                        alert('Session expired. Please login again.');
                        logout();
                    }
                } catch (error) {
                    addMessage('Error: Could not send message', 'bot');
                }
            }
            
            function addMessage(text, sender, intent = null) {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                
                let content = text;
                if (intent) {
                    content += `<br><span class="intent-badge ${intent}">${intent}</span>`;
                }
                
                messageDiv.innerHTML = content;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            async function loadChatHistory() {
                try {
                    const response = await fetch('/chat-history?limit=10', {
                        headers: {
                            'Authorization': `Bearer ${token}`,
                        }
                    });
                    
                    if (response.ok) {
                        const history = await response.json();
                        document.getElementById('messages').innerHTML = '';
                        history.forEach(item => {
                            addMessage(item.message, 'user');
                            addMessage(item.response, 'bot', item.intent);
                        });
                    }
                } catch (error) {
                    console.error('Could not load chat history');
                }
            }
            
            async function clearHistory() {
                if (!confirm('Are you sure you want to clear your chat history?')) {
                    return;
                }
                
                try {
                    const response = await fetch('/clear-history', {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${token}`,
                        }
                    });
                    
                    if (response.ok) {
                        document.getElementById('messages').innerHTML = '';
                        addMessage('Chat history cleared', 'bot');
                    }
                } catch (error) {
                    alert('Error clearing history');
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
