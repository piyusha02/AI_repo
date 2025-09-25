"""
Customer Sentiment Analyzer with SOR
Analyzes customer feedback with structured reasoning and actionable insights.
"""

import os
from typing import List, Literal, Dict
from pydantic import BaseModel, Field
from llm_config import get_llm_client, MODEL_NAME


class SentimentIndicator(BaseModel):
    """Specific indicator of sentiment"""
    phrase: str
    sentiment_impact: Literal["very_positive", "positive", "neutral", "negative", "very_negative"]
    category: Literal["product", "service", "price", "support", "delivery", "overall"]


class CustomerSentimentAnalysis(BaseModel):
    """Comprehensive sentiment analysis with business insights
    
    *** BUSINESS-ORIENTED REASONING CHAIN SOR PATTERN ***
    
    This schema demonstrates how SOR can be designed for business operations,
    connecting sentiment analysis to actionable business outcomes:
    
    BUSINESS REASONING FLOW:
    1. Sentiment assessment -> What is the customer feeling?
    2. Evidence extraction -> What specific phrases indicate this sentiment?
    3. Issue categorization -> What areas need attention?
    4. Churn risk evaluation -> Are we at risk of losing this customer?
    5. Operational response -> What should we do about it?
    
    BUSINESS INTELLIGENCE INTEGRATION:
    - Sentiment score enables aggregation and trending
    - Churn risk assessment drives retention strategies
    - Categorized aspects enable product/service improvement priorities
    - Response template selection enables automated customer service
    
    SOPHISTICATED FIELD DESIGN:
    - Nested SentimentIndicator model captures evidence with categorization
    - Separate emotion vs sentiment (angry customer might have valid points)
    - Churn risk separate from sentiment (disappointed ≠ leaving)
    - Actionable recommendations tied to specific business processes
    
    WHY THIS BUSINESS-FOCUSED PATTERN WORKS:
    1. Connects sentiment to business metrics (churn, satisfaction)
    2. Enables automated triage and response routing
    3. Provides structured feedback for product improvement
    4. Balances emotional analysis with operational needs
    5. Creates consistent customer service workflows
    
    FIELD DESIGN RATIONALE:
    - sentiment_score: float with constraints - enables trending/aggregation
    - key_sentiment_indicators: List[nested] - evidence-based analysis
    - Positive/negative aspects: balanced feedback extraction
    - churn_risk: business-critical customer retention metric
    - response_template_type: operational automation enabler
    
    PRO TIPS FOR BUSINESS SOR PATTERNS:
    - Connect analysis to specific business metrics (churn, satisfaction, NPS)
    - Separate emotional assessment from business risk assessment
    - Include both positive and negative feedback extraction
    - Design for automation integration (template selection, routing)
    - Provide evidence grounding for human review
    
    COMMON MISTAKES TO AVOID:
    - Don't assume negative sentiment = high churn risk (context matters)
    - Don't ignore positive aspects - miss opportunities for amplification
    - Don't make follow-up binary - use priority levels for resource allocation
    - Don't separate emotion from sentiment without good reason
    
    INTEGRATION WITH BUSINESS SYSTEMS:
    - Export high churn risk customers to retention campaigns
    - Route urgent follow-ups to senior customer success managers
    - Aggregate sentiment trends for product roadmap planning
    - Feed improvement suggestions into product backlog
    - Trigger automated response templates based on type classification
    
    PERFORMANCE OPTIMIZATION:
    - Sentiment score enables fast numerical sorting/filtering
    - Priority levels enable resource allocation optimization
    - Template selection reduces response time
    - Evidence extraction enables audit trail for service quality
    
    TESTING STRATEGIES:
    - Test with mixed sentiment (positive product, negative service)
    - Verify churn risk calibration across customer tenure
    - Check response template selection appropriateness
    - Validate improvement suggestion extraction quality
    
    MULTILINGUAL FIELD PATTERN:
    ===============================================
    This example demonstrates the SOR MULTILINGUAL PATTERN using a Spanish executive summary field.
    
    WHY MULTILINGUAL FIELDS IN SOR:
    1. Global business operations require multilingual support
    2. Executive summaries often need native language for stakeholders
    3. Customer service teams may be distributed globally
    4. Regulatory compliance may require local language documentation
    
    IMPLEMENTATION APPROACH:
    - Field name in English: 'executive_summary_spanish' (for code clarity)
    - Description in target language: Spanish instructions for the LLM
    - Constrained output: "Máximo 3 oraciones" (max 3 sentences)
    - Business focus: Include sentiment, churn risk, and actions
    
    BEST PRACTICES FOR MULTILINGUAL SOR:
    1. Use clear field names indicating target language (e.g., executive_summary_spanish)
    2. Provide instructions in the target language (Spanish description for Spanish output)
    3. Set clear constraints (sentence/word limits)
    4. Include key business metrics in summary
    5. Consider cultural context in language generation
    
    ADVANCED MULTILINGUAL PATTERNS:
    - Multiple language summaries in one schema
    - Language detection with appropriate response
    - Cultural adaptation of recommendations
    - Region-specific compliance fields
    
    TESTING MULTILINGUAL FIELDS:
    - Verify language quality with native speakers
    - Check for cultural appropriateness
    - Validate business term translations
    - Test with various regional dialects
    
    PRO TIP: LLMs handle multilingual fields remarkably well when:
    - Field names clearly indicate the target language (e.g., _spanish, _french)
    - Descriptions are written in the target language
    - Context makes the language requirement obvious
    
    This pattern enables global scalability of SOR systems!
    """
    
    # Overall sentiment assessment - primary business metric
    overall_sentiment: Literal["very_positive", "positive", "neutral", "negative", "very_negative"] = Field(
        description="Overall customer sentiment"
    )
    
    # Numerical score for aggregation and trending
    sentiment_score: float = Field(
        description="Numerical sentiment score from -1.0 to 1.0",
        ge=-1.0,
        le=1.0
    )
    
    # Evidence-based analysis with categorization
    key_sentiment_indicators: List[SentimentIndicator] = Field(
        description="Specific phrases and their sentiment impact"
    )
    
    # Balanced feedback extraction for product improvement
    positive_aspects: List[str] = Field(
        description="What the customer liked"
    )
    
    negative_aspects: List[str] = Field(
        description="What the customer disliked"
    )
    
    improvement_suggestions: List[str] = Field(
        description="Customer's suggestions for improvement"
    )
    
    # Emotional state assessment - separate from business risk
    emotion_detected: Literal["delighted", "satisfied", "neutral", "frustrated", "angry"] = Field(
        description="Primary emotion expressed"
    )
    
    # Business-critical customer retention metric
    churn_risk: Literal["low", "medium", "high"] = Field(
        description="Risk of losing this customer"
    )
    
    # Operational triage decision
    requires_followup: bool = Field(
        description="Whether this feedback needs immediate attention"
    )
    
    # Resource allocation guidance
    followup_priority: Literal["urgent", "high", "medium", "low"] = Field(
        description="Priority level for follow-up"
    )
    
    # Specific business actions - operational automation
    recommended_actions: List[str] = Field(
        description="Specific actions to address this feedback"
    )
    
    # Customer service automation enabler
    response_template_type: Literal["apology", "thank_you", "clarification", "resolution"] = Field(
        description="Type of response template to use"
    )
    
    # Multilingual support - Spanish executive summary
    executive_summary_spanish: str = Field(
        description="Proporcione un resumen ejecutivo en español del feedback del cliente, incluyendo el sentimiento principal, riesgo de pérdida del cliente, y acciones recomendadas. Máximo 3 oraciones."
    )


def analyze_customer_sentiment(feedback: str) -> CustomerSentimentAnalysis:
    """Analyze customer feedback for sentiment and actionable insights
    
    BUSINESS-ORIENTED SOR IMPLEMENTATION:
    - Customer experience analyst role for business context
    - Systematic analysis process connecting sentiment to business actions
    - Subtle cue identification for accurate risk assessment
    - Temperature 0.3 balances consistency with natural language nuance
    
    CUSTOMER SUCCESS INTEGRATION:
    - Churn risk assessment enables proactive retention
    - Priority classification drives resource allocation
    - Response template selection enables rapid customer service
    - Improvement suggestions feed product development cycle
    
    ADVANCED PATTERN RECOGNITION:
    - Extreme language -> high emotion detection
    - Multiple issues -> elevated churn risk
    - Constructive criticism -> engagement indicators
    - Sarcasm detection -> masked frustration identification
    """
    
    client = get_llm_client()
    
    completion = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": """You are a customer experience analyst.
                Analyze customer feedback by:
                1. Determining overall sentiment and emotion
                2. Identifying specific positive and negative indicators
                3. Assessing churn risk based on language intensity
                4. Extracting improvement suggestions
                5. Recommending concrete actions
                
                BUSINESS-ORIENTED SENTIMENT ANALYSIS:
                
                CHURN RISK ASSESSMENT:
                - HIGH: Explicit threats to leave, comparison shopping mentions, "final straw" language
                - MEDIUM: Frustration with multiple issues, questioning value, delayed responses
                - LOW: Single issue complaints, constructive feedback, long-term customer language
                
                PRIORITY CLASSIFICATION:
                - URGENT: Angry customers, service failures, public complaint threats
                - HIGH: Dissatisfied customers with specific issues, competitive comparisons
                - MEDIUM: Mixed feedback, process improvement suggestions
                - LOW: Minor issues, general feedback, satisfied customers with suggestions
                
                RESPONSE TEMPLATE SELECTION:
                - APOLOGY: Service failures, mistakes, unmet expectations
                - THANK_YOU: Positive feedback, compliments, constructive suggestions
                - CLARIFICATION: Misunderstandings, unclear processes, feature questions
                - RESOLUTION: Specific problems requiring concrete fixes
                
                Look for subtle cues:
                - Extreme language indicates high emotion
                - Multiple issues suggest high churn risk
                - Constructive criticism shows engaged customers
                - Sarcasm often masks frustration
                - Loyalty indicators ("long-time customer", "usually satisfied")"""
            },
            {
                "role": "user",
                "content": f"Analyze this customer feedback:\n{feedback}"
            }
        ],
        response_format=CustomerSentimentAnalysis,
        temperature=0.3  # Balance consistency with natural language nuance
    )
    
    return completion.choices[0].message.parsed


# Example usage
if __name__ == "__main__":
    customer_feedback = """
    I've been a customer for 3 years, but lately I'm really disappointed. 
    The product quality is still good, I'll give you that. But your customer service 
    has become absolutely terrible! I waited 45 minutes on hold yesterday just to 
    ask a simple question about my billing. 
    
    When I finally got through, the agent seemed like they didn't care at all and 
    couldn't even answer my question properly. They just kept reading from a script.
    
    The price increases don't help either - 20% more than last year for the same service? 
    That's ridiculous. I'm seriously considering switching to your competitor.
    
    Please do something about your support team. Train them better or hire people who 
    actually care about customers. Otherwise you'll lose a loyal customer.
    """
    
    result = analyze_customer_sentiment(customer_feedback)
    
    print(f"Overall Sentiment: {result.overall_sentiment} (Score: {result.sentiment_score:.2f})")
    print(f"Emotion Detected: {result.emotion_detected}")
    print(f"Churn Risk: {result.churn_risk}")
    print(f"Requires Follow-up: {'Yes' if result.requires_followup else 'No'} (Priority: {result.followup_priority})")
    
    print("\nPositive Aspects:")
    for aspect in result.positive_aspects:
        print(f"  ✓ {aspect}")
    
    print("\nNegative Aspects:")
    for aspect in result.negative_aspects:
        print(f"  ✗ {aspect}")
    
    print("\nKey Indicators:")
    for indicator in result.key_sentiment_indicators[:3]:  # Show top 3
        print(f"  • \"{indicator.phrase}\" → {indicator.sentiment_impact} ({indicator.category})")
    
    print("\nRecommended Actions:")
    for i, action in enumerate(result.recommended_actions, 1):
        print(f"  {i}. {action}")
    
    print(f"\nResponse Type: {result.response_template_type}")
    
    print("\n📝 Executive Summary (Spanish):")
    print(f"  {result.executive_summary_spanish}")
