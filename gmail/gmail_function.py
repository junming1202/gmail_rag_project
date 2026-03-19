# If modifying these scopes, delete the file token.json.
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os
import base64
from datetime import datetime

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def login_gmail(credential_path, token_path = None):
    creds = None
    # token.json stores the user's access and refresh tokens
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credential_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    return service

def get_simplified_message(service, msg_id):
    # Fetch message with 'full' format to get headers and body
    message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    
    payload = message.get('payload', {})
    headers = payload.get('headers', [])
    
    # 1. Extract Title (Subject) and Sender (From)
    title = "No Subject"
    sender = "Unknown Sender"
    
    for h in headers:
        if h['name'] == 'Subject':
            title = h['value']
        if h['name'] == 'From':
            sender = h['value']
    
    # 2. Extract and format Date/Time
    # internalDate is in milliseconds (UTC)
    timestamp = int(message.get('internalDate')) / 1000
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_date = dt_object.strftime('%Y-%m-%d %H:%M:%S')

    # 3. Extract Body (Plain Text only, ignoring images/HTML)
    def find_plain_text(parts):
        for part in parts:
            mime_type = part.get('mimeType')
            # Only grab the plain text part
            if mime_type == 'text/plain':
                data = part.get('body', {}).get('data')
                return base64.urlsafe_b64decode(data).decode('utf-8') if data else ""
            # Recursively search in nested parts (common in complex emails)
            if 'parts' in part:
                res = find_plain_text(part['parts'])
                if res: return res
        return ""

    if 'parts' in payload:
        body_text = find_plain_text(payload['parts'])
    else:
        # Fallback for simple emails that aren't multipart
        data = payload.get('body', {}).get('data')
        body_text = base64.urlsafe_b64decode(data).decode('utf-8') if data else ""

    # 4. Return the specific JSON-style dictionary
    return {
        "title": title,
        "sender": sender,
        "date_received": formatted_date,
        "body": body_text.strip()
    }

def get_emails_list(service, days=7):
    query = f"newer_than:{days}d"
    results = service.users().messages().list(userId='me', q=query).execute()
    messages = results.get('messages', [])
    
    cleaned_data = []
    
    for m in messages:
        # Call the transformation function
        simple_msg = get_simplified_message(service, m['id'])
        cleaned_data.append(simple_msg)
        
    return cleaned_data