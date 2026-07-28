# Configuration file for Iron Arrow Chatbot

import pandas as pd
import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API CONFIGURATION
# ============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

CLAUDE_MODEL = "claude-sonnet-4-6"

# Claude API settings

# Set thinking = {"type": "adaptive"}, output_config={"effort": CLAUDE_EFFORT}
CLAUDE_EFFORT = "high"  # Effort levels: low, medium, high, xhigh, max

# ============================================================================
# TOKEN LIMITS
# ============================================================================

MAX_TOKENS_PER_CONVERSATION = 50000
MAX_TOKENS_PER_REQUEST = 2500

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = "logs"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development or production

# ============================================================================
# RESOURCE FILE CONFIGURATION
# ============================================================================

#RESOURCES_FILE = "iron_arrow_resources.xlsx" # should contain metadata from iron arrow history pdf

# ============================================================================
# RESOURCE LOADING FROM EXCEL
# ============================================================================

# NOTE: can keep this the same if i use the same excel format

def load_resources_from_excel(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Load resources from Excel file

    Expected columns: key, name, url, description
    """
    try:
        # Read Excel file
        df = pd.read_excel(filepath)

        # Validate required columns
        required_columns = {'key', 'name', 'url', 'description'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Excel file missing required columns: {missing}")

        # Convert to dictionary format
        resources = {}
        for _, row in df.iterrows():
            # Skip empty rows
            if pd.isna(row['key']) or pd.isna(row['url']):
                continue

            key = str(row['key']).strip()
            resources[key] = {
                'name': str(row['name']).strip(),
                'url': str(row['url']).strip(),
                'description': str(row['description']).strip() if pd.notna(row['description']) else ""
            }

        if not resources:
            raise ValueError(f"No valid resources found in {filepath}")

        return resources

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Resources file not found: {filepath}\n"
            f"Please create an Excel file with columns: key, name, url, description"
        )
    except Exception as e:
        raise ValueError(f"Error loading resources from {filepath}: {str(e)}")


# Load resources from Excel
#RESOURCES = load_resources_from_excel(RESOURCES_FILE)

# ============================================================================
# SYSTEM PROMPT TEMPLATE
# ============================================================================

# change the prompt when you get here; keep resources_list

SYSTEM_PROMPT_TEMPLATE = """

You are a knowledgeable assistant specialising in the history of Iron Arrow, \
the honour society of the University of Miami.

Your role is to provide accurate, factual information using only the Society's approved online resources. \
You have access to these resources through the fetch_resource tool. \
When answering questions, rely only on information retrieved from those resources.

Guidelines
----------
- Answer using ONLY the information in the provided passages.
- If the passages do not contain enough information, say so clearly.
- Cite the section and page number(s) when you reference a specific fact, \
  e.g. "(The Beginnings, p. 12)".
- Be concise but thorough.  Use Markdown for structure when helpful.
- Do not invent facts or fill gaps with outside knowledge.


**SCOPE - ANSWER ONLY QUESTIONS ABOUT:**
- History and traditions
- Membership and tapping process
- Organization and leadership
- Programs and initiatives
- Events and ceremonies
- News and announcements
- Official policies and procedures
- Information published on the official Iron Arrow resources


**OUT-OF-SCOPE QUESTIONS:**
Do not answer questions unrelated to the Iron Arrow Honor Society or its official resources.

Examples include:

General University of Miami advising
Admissions
Financial aid
Housing
Academic advising
Legal or medical advice
Personal opinions or speculation

For out-of-scope questions, politely explain that you are the Iron Arrow Honor Society assistant and recommend \
contacting the appropriate office or organization.

If a question mixes in-scope and out-of-scope topics, respond only to the Iron Arrow portion if it can be answered \
independently. Otherwise, politely redirect the user.


**IMPORTANT:**
- If a question mixes in-scope and out-of-scope topics, redirect the entire question using the appropriate out-of-scope response.
- Never give advice, recommendations, instructions, or examples for out-of-scope topics.
- Keep in-scope answers factual, clear, and professional.
- Maintain a friendly, supportive tone for in-scope questions only.

**IN-SCOPE QUESTIONS:**
1. Use `fetch_resource` to retrieve official information.
2. Include at least one direct quote with full URL: According to [resource], "[quote]" (Source: [URL])
3. If information is missing: "I don't have that information in my current resources. Please contact a representative from the Iron Arrow Honor Society."

**AVAILABLE RESOURCES:**
Based on the topic of the question, retrieve the information from up to three links from the link resource list. Do not retrieve information from more than three links. Use the category and description found next to each link to understand whether it would be beneficial or not to retrieve information from this resource. Only retrieve information from links that are related to the question.

{resources_list}

"""

# ============================================================================
# WELCOME MESSAGE
# ============================================================================

WELCOME_MESSAGE = """\
👋 **Iron Arrow History Chatbot**

Ask me anything about the history of Iron Arrow — its founding, \
traditions, ceremonies, key figures, and more.

I retrieve the most relevant passages from *Iron Arrow: A History* \
and use them to answer your question accurately.

**Example questions:**
- Who founded Iron Arrow and when?
- What is the significance of Osceola to Iron Arrow?
- Describe the tapping ceremony.
- How did Iron Arrow survive World War II?
"""

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {

    "token_limit": "⚠️ This conversation has reached the length limit. Please start a new conversation to continue.",
    "rate_limit": "⏰ The AI service is rate limited. Please wait a moment and try again.",
    "api_key": "🔑 There's an issue with the API configuration. Please contact the administrator.",
    "timeout": "⏱️ The request timed out. The webpage might be slow. Please try again.",
    "generic": "❌ An error occurred: {error}\n\n💡 Please try rephrasing your question or contact an academic advisor for assistance."
}

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate that all required configuration is present"""
    errors = []

    if not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is required")

    if not RESOURCES:
        errors.append("At least one resource must be configured")

    if MAX_TOKENS_PER_CONVERSATION <= 0:
        errors.append("MAX_TOKENS_PER_CONVERSATION must be positive")

    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True


# Validate on import
validate_config()
