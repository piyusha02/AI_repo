"""
Email to Task Parser with SOR
Converts unstructured emails into structured, actionable tasks.
"""

import os
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from llm_config import get_llm_client, MODEL_NAME


class EmailTask(BaseModel):
    """Structured task extracted from email content
    
    SOR PRINCIPLE ANALYSIS:
    Field ordering follows natural email processing:
    1. Metadata first (sender, date) - establish context
    2. Core content extraction (action_items) - the primary goal
    3. Priority assessment - business-critical classification
    4. Timeline extraction - operational planning
    5. Context summary - human-readable overview
    
    FIELD DESIGN RATIONALE:
    - sender: str (not email regex) - handles "John Smith" or "john@company.com"
    - action_items: List[str] - separates multiple tasks for better tracking
    - priority: Literal - prevents hallucinated urgency levels like "super_urgent"
    - deadline: Optional[str] - acknowledges many emails lack specific dates
    
    PRO TIPS:
    - List[str] for action_items creates individual trackable tasks
    - Optional deadline prevents forcing non-existent dates
    - Priority extraction looks for linguistic cues: "ASAP", "when you can"
    - Context field provides human-readable summary for task management UIs
    
    COMMON MISTAKES TO AVOID:
    - Don't use complex datetime parsing - keep deadline as string for flexibility
    - Don't over-segment action items - group related sub-tasks
    - Don't ignore implicit priorities - "when you get a chance" = low priority
    
    TOKEN USAGE OPTIMIZATION:
    - Constrained enums (Literal) reduce output variance
    - Optional fields save tokens when data isn't present
    - String dates avoid complex datetime serialization overhead
    
    INTEGRATION WITH REAL SYSTEMS:
    - Parse sender against corporate directory
    - Map priority to SLA response times
    - Convert deadline strings to structured dates downstream
    - Link action_items to project management tools
    
    TESTING STRATEGIES:
    - Test with urgent/non-urgent language patterns
    - Verify handling of implicit vs explicit deadlines
    - Check action item granularity (not too broad, not too specific)
    """
    
    # Email metadata - establish context first
    sender: str = Field(description="Email sender's name or email address")
    date: str = Field(description="Date when email was sent")
    
    # Core task information - primary extraction goal
    action_items: List[str] = Field(
        description="List of specific, actionable tasks mentioned in the email"
    )
    
    # Priority assessment - business-critical classification
    priority: Literal["high", "medium", "low"] = Field(
        description="Priority level based on urgency indicators in email"
    )
    
    # Optional deadline extraction - many emails lack specific dates
    deadline: Optional[str] = Field(
        None,
        description="Specific deadline mentioned in email, or None if not specified"
    )
    
    # Context for better understanding - human-readable overview
    context: str = Field(
        description="Brief summary of the email context for the tasks"
    )


def parse_email_to_tasks(email_content: str) -> EmailTask:
    """Parse email content into structured tasks
    
    SOR PROMPT ENGINEERING:
    - Explicit step-by-step instructions guide LLM reasoning
    - Verb list (review, send, prepare) helps identify actionable items
    - Priority keywords prevent subjective interpretations
    - Temperature 0.3 balances consistency with natural language flexibility
    
    PRACTICAL EXTENSIONS:
    - Add email thread context for better understanding
    - Implement sender authority weighting (CEO emails = higher priority)
    - Extract @mentions for automatic assignee detection
    - Parse CC/BCC for stakeholder identification
    
    EDGE CASE HANDLING:
    - Forward/reply chains need context preservation
    - Automated emails (no-reply addresses) need special handling
    - Multi-language emails require language detection
    """
    
    client = get_llm_client()
    
    completion = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": """You are a task extraction specialist. 
                Extract actionable tasks from emails by:
                1. Identifying the sender and date
                2. Finding all action items (look for verbs like: review, send, prepare, schedule, complete)
                3. Determining priority based on words like: urgent, ASAP, critical, when you can
                4. Extracting specific deadlines if mentioned
                5. Summarizing the context
                
                SOR GUIDELINES:
                - Make action items specific: "Review budget proposal" not "Review document"
                - Priority HIGH: urgent, ASAP, critical, deadline today/tomorrow
                - Priority MEDIUM: please do, needed by [specific date]
                - Priority LOW: when you can, when convenient, no rush
                
                Be specific and actionable in task descriptions."""
            },
            {
                "role": "user",
                "content": f"Extract tasks from this email:\n{email_content}"
            }
        ],
        response_format=EmailTask,
        temperature=0.3,  # Balance consistency with natural language processing
        max_completion_tokens=500  # Sufficient for typical email task extraction
    )
    
    return completion.choices[0].message.parsed


# Example usage
if __name__ == "__main__":
    sample_email = """
    From: sarah.johnson@company.com
    Date: Monday, March 15, 2024
    Subject: Q2 Planning - Action Required
    
    Hi Team,
    
    Hope you're all doing well. We need to wrap up our Q2 planning soon. 
    
    Could you please:
    - Review the attached budget proposal and provide feedback by EOD Wednesday
    - Schedule a team meeting for next week to discuss the roadmap
    - Send me your individual OKRs for Q2 (this is URGENT - needed by tomorrow!)
    
    Also, when you get a chance, please update the project dashboard with current status.
    
    Thanks,
    Sarah
    """
    
    result = parse_email_to_tasks(sample_email)
    
    print(f"Sender: {result.sender}")
    print(f"Date: {result.date}")
    print(f"Priority: {result.priority}")
    print(f"Deadline: {result.deadline}")
    print("\nAction Items:")
    for i, task in enumerate(result.action_items, 1):
        print(f"  {i}. {task}")
    print(f"\nContext: {result.context}")
