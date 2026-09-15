import os

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, text

from google import genai


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL")


if not DATABASE_URL:
    raise Exception("DATABASE_URL is not configured")

if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY is not configured")

if not FRONTEND_URL:
    raise Exception("FRONTEND_URL is not configured")


# ============================================================
# DATABASE
# ============================================================

engine = create_engine(DATABASE_URL)


# ============================================================
# GEMINI
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AIRequest(BaseModel):
    user_id: int
    prompt: str


# ============================================================
# GET USER DATA
# ============================================================

def get_user_data(user_id: int):

    with engine.connect() as connection:

        # ----------------------------------------------------
        # USER
        # ----------------------------------------------------

        user_result = connection.execute(
            text("""
                SELECT
                    id,
                    full_name,
                    email
                FROM users
                WHERE id = :user_id
            """),
            {
                "user_id": user_id
            }
        )

        user = user_result.mappings().first()

        if not user:
            return None


        # ----------------------------------------------------
        # EXPENSES
        # ----------------------------------------------------

        expense_result = connection.execute(
            text("""
                SELECT
                    id,
                    amount,
                    date,
                    category,
                    description,
                    user_id,
                    icon,
                    created_at,
                    updated_at
                FROM expenses
                WHERE user_id = :user_id
                ORDER BY date DESC, id DESC
            """),
            {
                "user_id": user_id
            }
        )

        expenses = [
            dict(row)
            for row in expense_result.mappings().all()
        ]


        # ----------------------------------------------------
        # INCOMES
        # ----------------------------------------------------

        income_result = connection.execute(
            text("""
                SELECT
                    id,
                    amount,
                    date,
                    source,
                    description,
                    user_id,
                    icon,
                    created_at,
                    updated_at
                FROM incomes
                WHERE user_id = :user_id
                ORDER BY date DESC, id DESC
            """),
            {
                "user_id": user_id
            }
        )

        incomes = [
            dict(row)
            for row in income_result.mappings().all()
        ]


        # ----------------------------------------------------
        # RETURN DATA
        # ----------------------------------------------------

        return {
            "user": dict(user),
            "expenses": expenses,
            "incomes": incomes
        }


# ============================================================
# AI ANALYZE ENDPOINT
# ============================================================

@app.post("/ai/analyze")
def analyze_transactions(request: AIRequest):

    # --------------------------------------------------------
    # 1. Validate prompt
    # --------------------------------------------------------

    prompt = request.prompt.strip()

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Prompt is required"
        )


    # --------------------------------------------------------
    # 2. Fetch user data
    # --------------------------------------------------------

    user_data = get_user_data(request.user_id)

    if not user_data:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )


    # --------------------------------------------------------
    # 3. Build Gemini prompt
    # --------------------------------------------------------

    input_prompt = f"""
You are a financial assistant.

Here is the user's financial data:

{user_data}


User's query:

{prompt}


Please analyze or respond helpfully in a clear, concise way.

Always show money amounts in Indian Rupees format (₹).

Only use the financial data provided above.
Do not invent or assume financial information that is not present in the data.
"""


    # --------------------------------------------------------
    # 4. Call Gemini
    # --------------------------------------------------------

    try:

        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=input_prompt
        )

        reply = response.text

        return {
            "reply": reply
        }

    except Exception as ai_error:

        print(
            f"AI service unavailable: {ai_error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to analyze transactions"
        )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI server is running"
    }