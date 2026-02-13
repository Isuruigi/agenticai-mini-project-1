"""
Prompt Templates for Q&A Generation
"""

QUESTION_GENERATION_PROMPT = """You are a financial analyst creating training questions from Uber's 2024 Annual Report.

Given this excerpt from the report:

{chunk}

Generate EXACTLY 10 questions covering these categories:
- 3 Hard Facts (specific numbers, dates, percentages, revenue figures)
- 4 Strategic Summaries (business models, risk factors, competitive advantages, growth strategies)
- 3 Stylistic/Creative (tone analysis, CEO messaging style, target audience insights)

Output format (numbered list):
1. [Question]
2. [Question]
...
10. [Question]

Make questions specific to the content provided. Hard Fact questions should ask for precise data points that can be verified."""

ANSWER_GENERATION_PROMPT = """You are answering questions about Uber's 2024 Annual Report.

Context from the report:
{chunk}

Question: {question}

Provide a concise, accurate answer based ONLY on the context above.
- If the answer is in the context, provide it clearly and specifically
- If the answer is NOT in the context, respond: "Information not available in this section."
- Keep answers to 2-3 sentences maximum
- Include specific numbers, dates, or data when available"""


QUESTION_CATEGORIES = {
    'Hard Fact': [
        'revenue', 'billion', 'million', 'percent', '%', 'growth', 'quarter', 'q1', 'q2', 'q3', 'q4',
        'how much', 'how many', 'what was the', 'date', 'year', '2024', '2023', 'increase', 'decrease'
    ],
    'Strategic': [
        'strategy', 'business model', 'competitive', 'advantage', 'risk', 'opportunity', 'market',
        'how does', 'what are', 'why', 'approach', 'plan', 'goal', 'challenge', 'strength'
    ],
    'Stylistic': [
        'tone', 'message', 'audience', 'communication', 'emphasize', 'style', 'framing',
        'describe', 'characterize', 'language', 'messaging'
    ]
}


def categorize_question(question: str) -> str:
    """
    Categorize a question based on keywords
    
    Args:
        question: The question text
        
    Returns:
        Category name: 'Hard Fact', 'Strategic', or 'Stylistic'
    """
    question_lower = question.lower()
    
    # Count matches for each category
    scores = {}
    for category, keywords in QUESTION_CATEGORIES.items():
        score = sum(1 for keyword in keywords if keyword in question_lower)
        scores[category] = score
    
    # Return category with highest score, default to Strategic
    if max(scores.values()) == 0:
        return 'Strategic'
    
    return max(scores, key=scores.get)
