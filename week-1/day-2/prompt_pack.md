# Prompt 1 - Easy: Single Field Extraction

Task:
Extract the person's name.

Text:
John Smith is a software engineer.

Return ONLY JSON.

Schema:

{
  "name": "string"
}

Expected format:

{
  "name": "John Smith"
}


# Prompt 2 - Easy: Multiple Field Extraction

Task:
Extract employee information.

Text:

Sarah Khan works at ABC Technologies.
She is a Software Engineer with 3 years experience.

Return ONLY JSON.

Schema:

{
  "name": "string",
  "company": "string",
  "role": "string",
  "experience_years": "integer"
}


# Prompt 3 - Easy/Medium: Classification

Task:
Classify customer feedback.

Text:

"The product arrived late but quality is excellent."

Return ONLY JSON.

Schema:

{
  "sentiment": "Positive | Negative | Neutral",
  "reason": "string"
}


# Prompt 4 - Medium: Nested JSON

Task:
Extract user profile.

Text:

Ali is a Python developer from Lahore.
He knows Django and AWS.

Return ONLY JSON.

Schema:

{
  "user": {
    "name": "string",
    "location": "string",
    "skills": [
      "string"
    ]
  }
}


# Prompt 5 - Medium: Reasoning

Task:

Calculate project completion status.

Data:

Project has 5 tasks.
3 are completed.

Return ONLY JSON.

Schema:

{
  "total_tasks": "integer",
  "completed_tasks": "integer",
  "remaining_tasks": "integer",
  "completion_percentage": "number"
}


# Prompt 6 - Medium: Multi Object Array

Task:

Extract employees.

Text:

John is a developer.
Sarah is a designer.
Ali is a tester.

Return ONLY JSON.

Schema:

{
  "employees": [
    {
      "name": "string",
      "role": "string"
    }
  ]
}


# Prompt 7 - Hard: Missing Information

Task:

Extract customer details.

Text:

Customer John placed an order.

Some information may be missing.

Return ONLY JSON.

Schema:

{
  "name": "string",
  "email": "string|null",
  "phone": "string|null"
}


# Prompt 8 - Hard: Ambiguous Input

Task:

Resolve ambiguity.

Text:

Alex contacted Sam about the project.
He approved the changes.

Return ONLY JSON.

Schema:

{
  "people": [
    {
      "name": "string",
      "role": "unknown|string"
    }
  ],
  "ambiguity": "string"
}


# Prompt 9 - Hard: Conflicting Instructions

Task:

Extract data.

Important:
Return ONLY JSON.
Do not provide explanation.

Text:

Company XYZ hired Maria as a developer.

Schema:

{
  "company": "string",
  "employee": {
    "name": "string",
    "role": "string"
  }
}


# Prompt 10 - Hard: Complex Agent Output

Task:

Create a project plan.

Requirement:

Build an AI chatbot for customer support.

Return ONLY JSON.

Schema:

{
  "project": {
    "name": "string",
    "goals": [
      "string"
    ],
    "risks": [
      {
        "risk": "string",
        "impact": "High|Medium|Low"
      }
    ],
    "timeline_weeks": "integer"
  }
}