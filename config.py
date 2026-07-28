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

    #if not RESOURCES:
        errors.append("At least one resource must be configured")

    if MAX_TOKENS_PER_CONVERSATION <= 0:
        errors.append("MAX_TOKENS_PER_CONVERSATION must be positive")

    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True


# Validate on import
validate_config()
