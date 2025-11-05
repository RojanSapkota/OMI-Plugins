from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import time
import os
import requests
from collections import defaultdict
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
from datetime import datetime, timedelta
import threading
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
api_key = os.getenv('GROQ_API_KEY')
ai_model = os.getenv('AI_MODEL')

if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is required")


client = Groq(api_key=api_key)

# OMI App credentials for notifications
omi_app_id = os.getenv('HEY_OMI_APP_ID')
omi_app_secret = os.getenv('HEY_OMI_APP_SECRET')

if not omi_app_id or not omi_app_secret:
    raise ValueError("HEY_OMI_APP_ID and HEY_OMI_APP_SECRET environment variables are required")

# Initialize FastAPI app and router
app = FastAPI(title="OMI AI Assistant", version="1.0.0")
router = APIRouter()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modify trigger phrases - now just looking for "omi" followed by a question
TRIGGER_PHRASES = ["omi"]  # Base trigger - just "omi"
QUESTION_AGGREGATION_TIME = 10  # seconds to wait for collecting the question


# Replace the message buffer with a class to better manage state
class MessageBuffer:
    def __init__(self):
        self.buffers = {}
        self.lock = threading.Lock()
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()

    def get_buffer(self, session_id):
        current_time = time.time()

        # Cleanup old sessions periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_old_sessions()

        with self.lock:
            if session_id not in self.buffers:
                self.buffers[session_id] = {
                    'messages': [],
                    'trigger_detected': False,
                    'trigger_time': 0,
                    'collected_question': [],
                    'response_sent': False,
                    'partial_trigger': False,
                    'partial_trigger_time': 0,
                    'last_activity': current_time,
                }
            else:
                self.buffers[session_id]['last_activity'] = current_time

        return self.buffers[session_id]

    def cleanup_old_sessions(self):
        current_time = time.time()
        with self.lock:
            expired_sessions = [
                session_id
                for session_id, data in self.buffers.items()
                if current_time - data['last_activity'] > 3600  # Remove sessions older than 1 hour
            ]
            for session_id in expired_sessions:
                del self.buffers[session_id]
            self.last_cleanup = current_time


# Replace the message_buffer defaultdict with our new class
message_buffer = MessageBuffer()

# Add cooldown tracking
notification_cooldowns = defaultdict(float)
NOTIFICATION_COOLDOWN = 15  # 15 seconds cooldown between notifications for each session


class WebhookRequest(BaseModel):
    session_id: str
    segments: List[Dict[str, Any]] = []
    uid: str = None


class WebhookResponse(BaseModel):
    status: str = "success"
    message: str = None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_groq_response(text):
    """Get response from Groq for the user's question"""
    try:
        logger.info(f"Sending question to Groq: {text}")

        completion = client.chat.completions.create(
            model=ai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are Omi, a helpful AI assistant. Provide clear, concise, and friendly responses in single paragraph.",
                },
                {"role": "user", "content": text},
            ],
            temperature=0.7,
            max_completion_tokens=50,
            top_p=1,
            #reasoning_effort="medium",
            stream=False,
            stop=None,
        )

        answer = completion.choices[0].message.content.strip()
        logger.info(f"Received response from Groq: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error getting Groq response: {str(e)}")
        return "I'm sorry, I encountered an error processing your request."


def send_omi_notification(uid: str, message: str):
    """Send notification using OMI's notifications endpoint"""
    try:
        url = f"https://api.omi.me/v2/integrations/{omi_app_id}/notification"
        headers = {"Authorization": f"Bearer {omi_app_secret}", "Content-Type": "application/json"}
        params = {"uid": uid, "message": message}

        logger.info(f"Sending notification to OMI for uid {uid}: {message}")

        response = requests.post(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()

        logger.info(f"Successfully sent notification to OMI for uid {uid}")
        return True
    except Exception as e:
        logger.error(f"Error sending notification to OMI: {str(e)}")
        return False


@router.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "status_code": "200",
        "status": "online",
        "message": "OMI AI Assistant is running",
    }


@router.post('/webhook')
async def webhook(request: WebhookRequest):
    #logger.info("Received webhook POST request")
    #logger.info(f"Received data: {request.dict()}")

    session_id = request.session_id
    uid = request.uid or session_id  # Use session_id as uid if uid is not provided
    logger.info(f"Processing request for session_id: {session_id}, uid: {uid}")

    if not session_id:
        logger.error("No session_id provided in request")
        raise HTTPException(status_code=400, detail="No session_id provided")

    current_time = time.time()
    buffer_data = message_buffer.get_buffer(session_id)
    segments = request.segments
    has_processed = False

    # Add debug logging
    logger.debug(f"Current buffer state for session {session_id}: {buffer_data}")

    # Check and handle cooldown
    last_notification_time = notification_cooldowns.get(session_id, 0)
    time_since_last_notification = current_time - last_notification_time

    # If cooldown has expired, reset it
    if time_since_last_notification >= NOTIFICATION_COOLDOWN:
        notification_cooldowns[session_id] = 0

    # Only check cooldown if we have a trigger and are about to process
    if (
        buffer_data['trigger_detected']
        and not buffer_data['response_sent']
        and time_since_last_notification < NOTIFICATION_COOLDOWN
    ):
        logger.info(f"Cooldown active. {NOTIFICATION_COOLDOWN - time_since_last_notification:.0f}s remaining")
        return WebhookResponse(status="success")

    # Process each segment
    for segment in segments:
        if not segment.get('text') or has_processed:
            continue

        text = segment['text'].lower().strip()
        logger.info(f"Processing text segment: '{text}'")

        # Check for trigger phrase "omi" at the start of text
        trigger_found = False
        for trigger in [t.lower() for t in TRIGGER_PHRASES]:
            # Look for "omi" as a word (with word boundaries)
            if text.startswith(trigger) or f" {trigger} " in f" {text} " or text.startswith(f"{trigger},") or text.startswith(f"{trigger} "):
                trigger_found = True
                logger.info(f"Trigger phrase 'omi' detected in session {session_id}")
                buffer_data['trigger_detected'] = True
                buffer_data['trigger_time'] = current_time
                buffer_data['collected_question'] = []
                buffer_data['response_sent'] = False
                buffer_data['partial_trigger'] = False

                # Extract any question part that comes after the trigger
                # Handle "omi," "omi." "omi " etc.
                parts = text.split(trigger, 1)
                if len(parts) > 1:
                    question_part = parts[1].strip().lstrip('.,!? ')
                    if question_part:
                        buffer_data['collected_question'].append(question_part)
                        logger.info(f"Collected question part from trigger: {question_part}")
                        
                        # If the question seems complete (has "?"), process immediately
                        if '?' in question_part:
                            logger.info(f"Complete question detected in same segment: {question_part}")
                            full_question = question_part.strip()
                            
                            response = get_groq_response(full_question)
                            logger.info(f"Got response from Groq: {response}")
                            
                            # Reset all states
                            buffer_data['trigger_detected'] = False
                            buffer_data['trigger_time'] = 0
                            buffer_data['collected_question'] = []
                            buffer_data['response_sent'] = True
                            buffer_data['partial_trigger'] = False
                            has_processed = True
                            notification_cooldowns[session_id] = current_time
                            
                            return {"message": response}
                break
        
        if trigger_found:
            continue

        # If trigger was detected, collect the question
        if buffer_data['trigger_detected'] and not buffer_data['response_sent'] and not has_processed:
            time_since_trigger = current_time - buffer_data['trigger_time']
            logger.info(f"Time since trigger: {time_since_trigger} seconds")
            
            # Reset if trigger is too old (more than 30 seconds)
            if time_since_trigger > 30:
                logger.info(f"Trigger too old ({time_since_trigger}s), resetting state")
                buffer_data['trigger_detected'] = False
                buffer_data['trigger_time'] = 0
                buffer_data['collected_question'] = []
                buffer_data['response_sent'] = False
                buffer_data['partial_trigger'] = False
                continue

            if time_since_trigger <= QUESTION_AGGREGATION_TIME:
                buffer_data['collected_question'].append(text)
                logger.info(f"Collecting question part: {text}")
                logger.info(f"Current collected question: {' '.join(buffer_data['collected_question'])}")

            # Check if we should process the question
            should_process = (
                (time_since_trigger > QUESTION_AGGREGATION_TIME and buffer_data['collected_question'])
                or (buffer_data['collected_question'] and '?' in text)
                or (time_since_trigger > QUESTION_AGGREGATION_TIME * 1.5)
            )

            if should_process and buffer_data['collected_question']:
                # Process question and send notification
                full_question = ' '.join(buffer_data['collected_question']).strip()
                if not full_question.endswith('?'):
                    full_question += '?'

                logger.info(f"Processing complete question: {full_question}")
                response = get_groq_response(full_question)
                logger.info(f"Got response from Groq: {response}")

                # Reset all states
                buffer_data['trigger_detected'] = False
                buffer_data['trigger_time'] = 0
                buffer_data['collected_question'] = []
                buffer_data['response_sent'] = True
                buffer_data['partial_trigger'] = False
                has_processed = True
                notification_cooldowns[session_id] = current_time

                # Return the message directly
                return {"message": response}

    # Return success if no response needed
    return WebhookResponse(status="success")


@router.get('/webhook/setup-status')
async def setup_status():
    try:
        # Always return true for setup status
        return {"is_setup_completed": True}
    except Exception as e:
        logger.error(f"Error checking setup status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/status')
async def status():
    return {"active_sessions": len(message_buffer.buffers), "uptime": time.time() - start_time}


# Add at the top of the file with other globals
start_time = time.time()

# Include the router in the app
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)