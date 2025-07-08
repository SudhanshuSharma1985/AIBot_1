from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from config import Config
import json

class WellnessChatbot:
    def __init__(self):
        if Config.USE_AZURE:
            self.llm = AzureChatOpenAI(
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                openai_api_key=Config.AZURE_OPENAI_API_KEY,
                deployment_name=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
                openai_api_version=Config.AZURE_OPENAI_API_VERSION,
                temperature=0.7
            )
        else:
            self.llm = ChatOpenAI(
                openai_api_key=Config.OPENAI_API_KEY,
                model_name="gpt-3.5-turbo",
                temperature=0.7
            )
        
        # Intent detection prompt
        self.intent_prompt = ChatPromptTemplate.from_template("""
        You are an intent classifier for a social and emotional wellness chatbot.
        Classify the user's message into one of these categories:
        
        1. "emergency": Messages indicating immediate danger, self-harm, suicide, or crisis situations
        2. "irrelevant": Messages not related to social and emotional wellness (e.g., technical questions, math, unrelated topics)
        3. "qna": Messages related to social and emotional wellness, mental health, relationships, stress, anxiety, etc.
        
        User message: {user_message}
        
        Respond with a JSON object containing:
        - "intent": one of ["emergency", "irrelevant", "qna"]
        - "confidence": a number between 0 and 1
        - "reason": brief explanation for the classification
        
        Response:
        """)
        
        # Response generation prompts
        self.emergency_prompt = ChatPromptTemplate.from_template("""
        You are a compassionate crisis counselor. The user has expressed something concerning that may indicate they are in crisis.
        
        User message: {user_message}
        
        Provide a supportive, non-judgmental response that:
        1. Acknowledges their feelings
        2. Expresses care and concern
        3. Provides crisis helpline numbers
        4. Encourages immediate professional help
        
        Important crisis resources to include:
        - National Suicide Prevention Lifeline: 988 or 1-800-273-8255
        - Crisis Text Line: Text HOME to 741741
        - Emergency services: 911
        
        Response:
        """)
        
        self.wellness_prompt = ChatPromptTemplate.from_template("""
        You are a supportive social and emotional wellness assistant. Provide helpful, empathetic responses about mental health, emotions, relationships, and wellbeing.
        
        User message: {user_message}
        
        Guidelines:
        - Be warm, empathetic, and non-judgmental
        - Provide practical advice and coping strategies
        - Encourage self-care and healthy habits
        - Suggest professional help when appropriate
        - Use evidence-based approaches
        
        Response:
        """)
        
        self.irrelevant_prompt = ChatPromptTemplate.from_template("""
        The user has asked something outside the scope of social and emotional wellness support.
        
        User message: {user_message}
        
        Politely redirect them to wellness-related topics. Suggest how you can help with:
        - Stress management
        - Emotional support
        - Relationship advice
        - Mental health resources
        - Self-care strategies
        
        Response:
        """)
    
    def detect_intent(self, user_message: str) -> dict:
        chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
        response = chain.run(user_message=user_message)
        
        try:
            intent_data = json.loads(response)
        except:
            # Fallback if JSON parsing fails
            intent_data = {
                "intent": "qna",
                "confidence": 0.5,
                "reason": "Could not parse intent, defaulting to Q&A"
            }
        
        return intent_data
    
    def generate_response(self, user_message: str, intent: str) -> str:
        if intent == "emergency":
            chain = LLMChain(llm=self.llm, prompt=self.emergency_prompt)
        elif intent == "irrelevant":
            chain = LLMChain(llm=self.llm, prompt=self.irrelevant_prompt)
        else:  # qna
            chain = LLMChain(llm=self.llm, prompt=self.wellness_prompt)
        
        return chain.run(user_message=user_message)