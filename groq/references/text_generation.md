# Groq - Text Generation

**Pages:** 13

---

## Code execution tool call

**URL:** llms-txt#code-execution-tool-call

**Contents:**
- Code Execution: Gpt Oss Quickstart (js)
- Code Execution

if response.choices[0].message.executed_tools:
    print(response.choices[0].message.executed_tools[0])

## Code Execution: Gpt Oss Quickstart (js)

URL: https://console.groq.com/docs/code-execution/scripts/gpt-oss-quickstart

import Groq from "groq-sdk";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const response = await groq.chat.completions.create({
  messages: [
    {
      role: "user",
      content: "Calculate the square root of 12345. Output only the final answer.",
    },
  ],
  model: "openai/gpt-oss-20b", // or "openai/gpt-oss-120b"
  tool_choice: "required",
  tools: [
    {
      type: "code_interpreter"
    },
  ],
});

// Final output
console.log(response.choices[0].message.content);

// Reasoning + internal tool calls
console.log(response.choices[0].message.reasoning);

// Code execution tool call
console.log(response.choices[0].message.executed_tools?.[0]);

URL: https://console.groq.com/docs/code-execution

---

## (example Python code)

**URL:** llms-txt#(example-python-code)

**Contents:**
  - Next Steps
- Structured Outputs: Email Classification (py)
- Structured Outputs: Sql Query Generation (js)
- Structured Outputs: File System Schema (json)
- Structured Outputs: Appointment Booking Schema (json)
- Structured Outputs: Task Creation Schema (json)
- Structured Outputs: Support Ticket Zod.doc (ts)
- Structured Outputs: Email Classification Response (json)
- Structured Outputs: Step2 Example (py)
- Structured Outputs: Api Response Validation (py)

javascript
import Groq from "groq-sdk";

const groq = new Groq();

const response = await groq.chat.completions.create({
  model: "moonshotai/kimi-k2-instruct-0905",
  messages: [
    {
      role: "system",
      content: "You are a SQL expert. Generate structured SQL queries from natural language descriptions with proper syntax validation and metadata.",
    },
    { role: "user", content: "Find all customers who made orders over $500 in the last 30 days, show their name, email, and total order amount" },
  ],
  response_format: {
    type: "json_schema",
    json_schema: {
      name: "sql_query_generation",
      schema: {
        type: "object",
        properties: {
          query: { type: "string" },
          query_type: { 
            type: "string", 
            enum: ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"] 
          },
          tables_used: {
            type: "array",
            items: { type: "string" }
          },
          estimated_complexity: {
            type: "string",
            enum: ["low", "medium", "high"]
          },
          execution_notes: {
            type: "array",
            items: { type: "string" }
          },
          validation_status: {
            type: "object",
            properties: {
              is_valid: { type: "boolean" },
              syntax_errors: {
                type: "array",
                items: { type: "string" }
              }
            },
            required: ["is_valid", "syntax_errors"],
            additionalProperties: false
          }
        },
        required: ["query", "query_type", "tables_used", "estimated_complexity", "execution_notes", "validation_status"],
        additionalProperties: false
      }
    }
  }
});

const result = JSON.parse(response.choices[0].message.content || "{}");
console.log(result);
javascript
import Groq from "groq-sdk";
import { z } from "zod";

const groq = new Groq();

const supportTicketSchema = z.object({
  category: z.enum(["api", "billing", "account", "bug", "feature_request", "integration", "security", "performance"]),
  priority: z.enum(["low", "medium", "high", "critical"]),
  urgency_score: z.number(),
  customer_info: z.object({
    name: z.string(),
    company: z.string().optional(),
    tier: z.enum(["free", "paid", "enterprise", "trial"])
  }),
  technical_details: z.array(z.object({
    component: z.string(),
    error_code: z.string().optional(),
    description: z.string()
  })),
  keywords: z.array(z.string()),
  requires_escalation: z.boolean(),
  estimated_resolution_hours: z.number(),
  follow_up_date: z.string().datetime().optional(),
  summary: z.string()
});

type SupportTicket = z.infer<typeof supportTicketSchema>;

const response = await groq.chat.completions.create({
  model: "moonshotai/kimi-k2-instruct-0905",
  messages: [
    {
      role: "system",
      content: `You are a customer support ticket classifier for SaaS companies. 
                Analyze support tickets and categorize them for efficient routing and resolution.
                Output JSON only using the schema provided.`,
    },
    { 
      role: "user", 
      content: `Hello! I love your product and have been using it for 6 months. 
                I was wondering if you could add a dark mode feature to the dashboard? 
                Many of our team members work late hours and would really appreciate this. 
                Also, it would be great to have keyboard shortcuts for common actions. 
                Not urgent, but would be a nice enhancement! 
                Best, Mike from StartupXYZ`
    },
  ],
  response_format: {
    type: "json_schema",
    json_schema: {
      name: "support_ticket_classification",
      schema: z.toJSONSchema(supportTicketSchema)
    }
  }
});

const rawResult = JSON.parse(response.choices[0].message.content || "{}");
const result = supportTicketSchema.parse(rawResult);
console.log(result);
json
{
  "category": "urgent",
  "priority": "critical",
  "confidence_score": 0.95,
  "sentiment": "negative",
  "key_entities": [
    {
      "entity": "production server",
      "type": "system"
    },
    {
      "entity": "2:30 PM EST",
      "type": "datetime"
    },
    {
      "entity": "DevOps Team",
      "type": "organization"
    },
    {
      "entity": "customer-facing services",
      "type": "system"
    }
  ],
  "suggested_actions": [
    "Join emergency call immediately",
    "Escalate to senior DevOps team",
    "Activate incident response protocol",
    "Prepare customer communication",
    "Monitor service restoration progress"
  ],
  "requires_immediate_attention": true,
  "estimated_response_time": "immediate"
}
python
from groq import Groq
from pydantic import BaseModel
import json

class ValidationResult(BaseModel):
    is_valid: bool
    status_code: int
    error_count: int

class FieldValidation(BaseModel):
    field_name: str
    field_type: str
    is_valid: bool
    error_message: str
    expected_format: str

class ComplianceCheck(BaseModel):
    follows_rest_standards: bool
    has_proper_error_handling: bool
    includes_metadata: bool

class Metadata(BaseModel):
    timestamp: str
    request_id: str
    version: str

class StandardizedResponse(BaseModel):
    success: bool
    data: dict
    errors: list[str]
    metadata: Metadata

class APIResponseValidation(BaseModel):
    validation_result: ValidationResult
    field_validations: list[FieldValidation]
    data_quality_score: float
    suggested_fixes: list[str]
    compliance_check: ComplianceCheck
    standardized_response: StandardizedResponse

response = client.chat.completions.create(
    model="moonshotai/kimi-k2-instruct-0905",
    messages=[
        {
            "role": "system",
            "content": "You are an API response validation expert. Validate and structure API responses with error handling, status codes, and standardized data formats for reliable integration.",
        },
        {"role": "user", "content": "Validate this API response: {\"user_id\": \"12345\", \"email\": \"invalid-email\", \"created_at\": \"2024-01-15T10:30:00Z\", \"status\": \"active\", \"profile\": {\"name\": \"John Doe\", \"age\": 25}}"},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "api_response_validation",
            "schema": APIResponseValidation.model_json_schema()
        }
    }
)

api_response_validation = APIResponseValidation.model_validate(json.loads(response.choices[0].message.content))
print(json.dumps(api_response_validation.model_dump(), indent=2))
javascript
import Groq from "groq-sdk";

const groq = new Groq();

const response = await groq.chat.completions.create({
  model: "moonshotai/kimi-k2-instruct-0905",
  messages: [
    {
      role: "system",
      content: "You are an API response validation expert. Validate and structure API responses with error handling, status codes, and standardized data formats for reliable integration.",
    },
    { role: "user", content: "Validate this API response: {\"user_id\": \"12345\", \"email\": \"invalid-email\", \"created_at\": \"2024-01-15T10:30:00Z\", \"status\": \"active\", \"profile\": {\"name\": \"John Doe\", \"age\": 25}}" },
  ],
  response_format: {
    type: "json_schema",
    json_schema: {
      name: "api_response_validation",
      schema: {
        type: "object",
        properties: {
          validation_result: {
            type: "object",
            properties: {
              is_valid: { type: "boolean" },
              status_code: { type: "integer" },
              error_count: { type: "integer" }
            },
            required: ["is_valid", "status_code", "error_count"],
            additionalProperties: false
          },
          field_validations: {
            type: "array",
            items: {
              type: "object",
              properties: {
                field_name: { type: "string" },
                field_type: { type: "string" },
                is_valid: { type: "boolean" },
                error_message: { type: "string" },
                expected_format: { type: "string" }
              },
              required: ["field_name", "field_type", "is_valid", "error_message", "expected_format"],
              additionalProperties: false
            }
          },
          data_quality_score: { 
            type: "number", 
            minimum: 0, 
            maximum: 1 
          },
          suggested_fixes: {
            type: "array",
            items: { type: "string" }
          },
          compliance_check: {
            type: "object",
            properties: {
              follows_rest_standards: { type: "boolean" },
              has_proper_error_handling: { type: "boolean" },
              includes_metadata: { type: "boolean" }
            },
            required: ["follows_rest_standards", "has_proper_error_handling", "includes_metadata"],
            additionalProperties: false
          },
          standardized_response: {
            type: "object",
            properties: {
              success: { type: "boolean" },
              data: { type: "object" },
              errors: {
                type: "array",
                items: { type: "string" }
              },
              metadata: {
                type: "object",
                properties: {
                  timestamp: { type: "string" },
                  request_id: { type: "string" },
                  version: { type: "string" }
                },
                required: ["timestamp", "request_id", "version"],
                additionalProperties: false
              }
            },
            required: ["success", "data", "errors", "metadata"],
            additionalProperties: false
          }
        },
        required: ["validation_result", "field_validations", "data_quality_score", "suggested_fixes", "compliance_check", "standardized_response"],
        additionalProperties: false
      }
    }
  }
});

const result = JSON.parse(response.choices[0].message.content || "{}");
console.log(result);
json
{
  "validation_result": {
    "is_valid": false,
    "status_code": 400,
    "error_count": 2
  },
  "field_validations": [
    {
      "field_name": "user_id",
      "field_type": "string",
      "is_valid": true,
      "error_message": "",
      "expected_format": "string"
    },
    {
      "field_name": "email",
      "field_type": "string",
      "is_valid": false,
      "error_message": "Invalid email format",
      "expected_format": "valid email address (e.g., user@example.com)"
    },
    {
      "field_name": "created_at",
      "field_type": "string",
      "is_valid": true,
      "error_message": "",
      "expected_format": "ISO 8601 datetime string"
    },
    {
      "field_name": "status",
      "field_type": "string",
      "is_valid": true,
      "error_message": "",
      "expected_format": "string"
    },
    {
      "field_name": "profile",
      "field_type": "object",
      "is_valid": true,
      "error_message": "",
      "expected_format": "object"
    }
  ],
  "data_quality_score": 0.7,
  "suggested_fixes": [
    "Fix email format validation to ensure proper email structure",
    "Add proper error handling structure to response",
    "Include metadata fields like timestamp and request_id",
    "Add success/failure status indicators",
    "Implement standardized error format"
  ],
  "compliance_check": {
    "follows_rest_standards": false,
    "has_proper_error_handling": false,
    "includes_metadata": false
  },
  "standardized_response": {
    "success": false,
    "data": {
      "user_id": "12345",
      "email": "invalid-email",
      "created_at": "2024-01-15T10:30:00Z",
      "status": "active",
      "profile": {
        "name": "John Doe",
        "age": 25
      }
    },
    "errors": [
      "Invalid email format: invalid-email",
      "Response lacks proper error handling structure"
    ],
    "metadata": {
      "timestamp": "2024-01-15T10:30:00Z",
      "request_id": "req_12345",
      "version": "1.0"
    }
  }
}
python
from groq import Groq
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum
import json

class SupportCategory(str, Enum):
    API = "api"
    BILLING = "billing"
    ACCOUNT = "account"
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    INTEGRATION = "integration"
    SECURITY = "security"
    PERFORMANCE = "performance"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CustomerTier(str, Enum):
    FREE = "free"
    PAID = "paid"
    ENTERPRISE = "enterprise"
    TRIAL = "trial"

class CustomerInfo(BaseModel):
    name: str
    company: Optional[str] = None
    tier: CustomerTier

class TechnicalDetail(BaseModel):
    component: str
    error_code: Optional[str] = None
    description: str

class SupportTicket(BaseModel):
    category: SupportCategory
    priority: Priority
    urgency_score: float
    customer_info: CustomerInfo
    technical_details: List[TechnicalDetail]
    keywords: List[str]
    requires_escalation: bool
    estimated_resolution_hours: float
    follow_up_date: Optional[str] = Field(None, description="ISO datetime string")
    summary: str

response = client.chat.completions.create(
    model="moonshotai/kimi-k2-instruct-0905",
    messages=[
        {
            "role": "system",
            "content": """You are a customer support ticket classifier for SaaS companies. 
                         Analyze support tickets and categorize them for efficient routing and resolution.
                         Output JSON only using the schema provided.""",
        },
        { 
            "role": "user", 
            "content": """Hello! I love your product and have been using it for 6 months. 
                         I was wondering if you could add a dark mode feature to the dashboard? 
                         Many of our team members work late hours and would really appreciate this. 
                         Also, it would be great to have keyboard shortcuts for common actions. 
                         Not urgent, but would be a nice enhancement! 
                         Best, Mike from StartupXYZ"""
        },
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "support_ticket_classification",
            "schema": SupportTicket.model_json_schema()
        }
    }
)

raw_result = json.loads(response.choices[0].message.content or "{}")
result = SupportTicket.model_validate(raw_result)
print(result.model_dump_json(indent=2))
python
from groq import Groq
from pydantic import BaseModel
import json

class ValidationStatus(BaseModel):
    is_valid: bool
    syntax_errors: list[str]

class SQLQueryGeneration(BaseModel):
    query: str
    query_type: str
    tables_used: list[str]
    estimated_complexity: str
    execution_notes: list[str]
    validation_status: ValidationStatus

response = client.chat.completions.create(
    model="moonshotai/kimi-k2-instruct-0905",
    messages=[
        {
            "role": "system",
            "content": "You are a SQL expert. Generate structured SQL queries from natural language descriptions with proper syntax validation and metadata.",
        },
        {"role": "user", "content": "Find all customers who made orders over $500 in the last 30 days, show their name, email, and total order amount"},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "sql_query_generation",
            "schema": SQLQueryGeneration.model_json_schema()
        }
    }
)

sql_query_generation = SQLQueryGeneration.model_validate(json.loads(response.choices[0].message.content))
print(json.dumps(sql_query_generation.model_dump(), indent=2))

{
  "type": "object",
  "properties": {
    "milestones": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/milestone"
      }
    },
    "project_status": {
      "type": "string",
      "enum": ["planning", "in_progress", "completed", "on_hold"]
    }
  },
  "$defs": {
    "milestone": {
      "type": "object",
      "properties": {
        "title": {
          "type": "string",
          "description": "Milestone name"
        },
        "deadline": {
          "type": "string",
          "description": "Due date in ISO format"
        },
        "completed": {
          "type": "boolean"
        }
      },
      "required": ["title", "deadline", "completed"],
      "additionalProperties": false
    }
  },
  "required": ["milestones", "project_status"],
  "additionalProperties": false
}
javascript
import { Groq } from "groq-sdk";

const groq = new Groq();

async function main() {
  const response = await groq.chat.completions.create({
    model: "openai/gpt-oss-20b",
    messages: [
      {
        role: "system",
        content: `You are a data analysis API that performs sentiment analysis on text.
                Respond only with JSON using this format:
                {
                    "sentiment_analysis": {
                    "sentiment": "positive|negative|neutral",
                    "confidence_score": 0.95,
                    "key_phrases": [
                        {
                        "phrase": "detected key phrase",
                        "sentiment": "positive|negative|neutral"
                        }
                    ],
                    "summary": "One sentence summary of the overall sentiment"
                    }
                }`
      },
      { role: "user", content: "Analyze the sentiment of this customer review: 'I absolutely love this product! The quality exceeded my expectations, though shipping took longer than expected.'" }
    ],
    response_format: { type: "json_object" }
  });

const result = JSON.parse(response.choices[0].message.content || "{}");
  console.log(result);
}

main();
javascript
import Groq from "groq-sdk";

const groq = new Groq();

const response = await groq.chat.completions.create({
  model: "moonshotai/kimi-k2-instruct-0905",
  messages: [
    { role: "system", content: "Extract product review information from the text." },
    {
      role: "user",
      content: "I bought the UltraSound Headphones last week and I'm really impressed! The noise cancellation is amazing and the battery lasts all day. Sound quality is crisp and clear. I'd give it 4.5 out of 5 stars.",
    },
  ],
  response_format: {
    type: "json_schema",
    json_schema: {
      name: "product_review",
      schema: {
        type: "object",
        properties: {
          product_name: { type: "string" },
          rating: { type: "number" },
          sentiment: { 
            type: "string",
            enum: ["positive", "negative", "neutral"]
          },
          key_features: { 
            type: "array",
            items: { type: "string" }
          }
        },
        required: ["product_name", "rating", "sentiment", "key_features"],
        additionalProperties: false
      }
    }
  }
});

const result = JSON.parse(response.choices[0].message.content || "{}");
console.log(result);
javascript
import Groq from "groq-sdk";

const groq = new Groq();

const response = await groq.chat.completions.create({
  model: "moonshotai/kimi-k2-instruct-0905",
  messages: [
    {
      role: "system",
      content: "You are an email classification expert. Classify emails into structured categories with confidence scores, priority levels, and suggested actions.",
    },
    { role: "user", content: "Subject: URGENT: Server downtime affecting production\n\nHi Team,\n\nOur main production server went down at 2:30 PM EST. Customer-facing services are currently unavailable. We need immediate action to restore services. Please join the emergency call.\n\nBest regards,\nDevOps Team" },
  ],
  response_format: {
    type: "json_schema",
    json_schema: {
      name: "email_classification",
      schema: {
        type: "object",
        properties: {
          category: { 
            type: "string", 
            enum: ["urgent", "support", "sales", "marketing", "internal", "spam", "notification"] 
          },
          priority: { 
            type: "string", 
            enum: ["low", "medium", "high", "critical"] 
          },
          confidence_score: { 
            type: "number", 
            minimum: 0, 
            maximum: 1 
          },
          sentiment: { 
            type: "string", 
            enum: ["positive", "negative", "neutral"] 
          },
          key_entities: {
            type: "array",
            items: {
              type: "object",
              properties: {
                entity: { type: "string" },
                type: { 
                  type: "string", 
                  enum: ["person", "organization", "location", "datetime", "system", "product"] 
                }
              },
              required: ["entity", "type"],
              additionalProperties: false
            }
          },
          suggested_actions: {
            type: "array",
            items: { type: "string" }
          },
          requires_immediate_attention: { type: "boolean" },
          estimated_response_time: { type: "string" }
        },
        required: ["category", "priority", "confidence_score", "sentiment", "key_entities", "suggested_actions", "requires_immediate_attention", "estimated_response_time"],
        additionalProperties: false
      }
    }
  }
});

const result = JSON.parse(response.choices[0].message.content || "{}");
console.log(result);

{
  "type": "object",
  "properties": {
    "payment_method": {
      "anyOf": [
        {
          "type": "object",
          "description": "Credit card payment information",
          "properties": {
            "card_number": {
              "type": "string",
              "description": "The credit card number"
            },
            "expiry_date": {
              "type": "string",
              "description": "Card expiration date in MM/YY format"
            },
            "cvv": {
              "type": "string",
              "description": "Card security code"
            }
          },
          "additionalProperties": false,
          "required": ["card_number", "expiry_date", "cvv"]
        },
        {
          "type": "object",
          "description": "Bank transfer payment information",
          "properties": {
            "account_number": {
              "type": "string",
              "description": "Bank account number"
            },
            "routing_number": {
              "type": "string",
              "description": "Bank routing number"
            },
            "bank_name": {
              "type": "string",
              "description": "Name of the bank"
            }
          },
          "additionalProperties": false,
          "required": ["account_number", "routing_number", "bank_name"]
        }
      ]
    }
  },
  "additionalProperties": false,
  "required": ["payment_method"]
}
javascript
import Groq from "groq-sdk";

const groq = new Groq();

const response = await groq.chat.completions.create({
    model: "moonshotai/kimi-k2-instruct-0905",
    messages: [
        { role: "system", content: "You are a helpful math tutor. Guide the user through the solution step by step." },
        { role: "user", content: "how can I solve 8x + 7 = -23" }
    ],
    response_format: {
        type: "json_schema",
        json_schema: {
            name: "math_response",
            schema: {
                type: "object",
                properties: {
                    steps: {
                        type: "array",
                        items: {
                            type: "object",
                            properties: {
                                explanation: { type: "string" },
                                output: { type: "string" }
                            },
                            required: ["explanation", "output"],
                            additionalProperties: false
                        }
                    },
                    final_answer: { type: "string" }
                },
                required: ["steps", "final_answer"],
                additionalProperties: false
            }
        }
    }
});

const result = JSON.parse(response.choices[0].message.content || "{}");
console.log(result);
json
{
  "name": "organization_chart",
  "description": "Company organizational structure",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "employee_id": {
        "type": "string",
        "description": "Unique employee identifier"
      },
      "name": {
        "type": "string",
        "description": "Employee full name"
      },
      "position": {
        "type": "string",
        "description": "Job title or position",
        "enum": ["CEO", "Manager", "Developer", "Designer", "Analyst", "Intern"]
      },
      "direct_reports": {
        "type": "array",
        "description": "Employees reporting to this person",
        "items": {
          "$ref": "#"
        }
      },
      "contact_info": {
        "type": "array",
        "description": "Contact information for the employee",
        "items": {
          "type": "object",
          "properties": {
            "type": {
              "type": "string",
              "description": "Type of contact info",
              "enum": ["email", "phone", "slack"]
            },
            "value": {
              "type": "string",
              "description": "The contact value"
            }
          },
          "additionalProperties": false,
          "required": ["type", "value"]
        }
      }
    },
    "required": [
      "employee_id",
      "name",
      "position",
      "direct_reports",
      "contact_info"
    ],
    "additionalProperties": false
  }
}
```

## Structured Outputs

URL: https://console.groq.com/docs/structured-outputs

**Examples:**

Example 1 (unknown):
```unknown
Now that you have successfully received a chat completion, you can try out the other endpoints in the API.

### Next Steps

- Check out the [Playground](/playground) to try out the Groq API in your browser
- Join our GroqCloud [developer community](https://community.groq.com/)
- Add a how-to on your project to the [Groq API Cookbook](https://github.com/groq/groq-api-cookbook)

---

## Structured Outputs: Email Classification (py)

URL: https://console.groq.com/docs/structured-outputs/scripts/email-classification.py

from groq import Groq
from pydantic import BaseModel
import json

client = Groq()

class KeyEntity(BaseModel):
    entity: str
    type: str

class EmailClassification(BaseModel):
    category: str
    priority: str
    confidence_score: float
    sentiment: str
    key_entities: list[KeyEntity]
    suggested_actions: list[str]
    requires_immediate_attention: bool
    estimated_response_time: str

response = client.chat.completions.create(
    model="moonshotai/kimi-k2-instruct-0905",
    messages=[
        {
            "role": "system",
            "content": "You are an email classification expert. Classify emails into structured categories with confidence scores, priority levels, and suggested actions.",
        },
        {"role": "user", "content": "Subject: URGENT: Server downtime affecting production\\n\\nHi Team,\\n\\nOur main production server went down at 2:30 PM EST. Customer-facing services are currently unavailable. We need immediate action to restore services. Please join the emergency call.\\n\\nBest regards,\\nDevOps Team"},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "email_classification",
            "schema": EmailClassification.model_json_schema()
        }
    }
)

email_classification = EmailClassification.model_validate(json.loads(response.choices[0].message.content))
print(json.dumps(email_classification.model_dump(), indent=2))

---

## Structured Outputs: Sql Query Generation (js)

URL: https://console.groq.com/docs/structured-outputs/scripts/sql-query-generation
```

Example 2 (unknown):
```unknown
---

## Structured Outputs: File System Schema (json)

URL: https://console.groq.com/docs/structured-outputs/scripts/file-system-schema.json

{
  "type": "object",
  "properties": {
    "file_system": {
      "$ref": "#/$defs/file_node"
    }
  },
  "$defs": {
    "file_node": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "description": "File or directory name"
        },
        "type": {
          "type": "string",
          "enum": ["file", "directory"]
        },
        "size": {
          "type": "number",
          "description": "Size in bytes (0 for directories)"
        },
        "children": {
          "anyOf": [
            {
              "type": "array",
              "items": {
                "$ref": "#/$defs/file_node"
              }
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "additionalProperties": false,
      "required": ["name", "type", "size", "children"]
    }
  },
  "additionalProperties": false,
  "required": ["file_system"]
}

---

## Structured Outputs: Appointment Booking Schema (json)

URL: https://console.groq.com/docs/structured-outputs/scripts/appointment-booking-schema.json

{
  "name": "book_appointment",
  "description": "Books a medical appointment",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "patient_name": {
        "type": "string",
        "description": "Full name of the patient"
      },
      "appointment_type": {
        "type": "string",
        "description": "Type of medical appointment",
        "enum": ["consultation", "checkup", "surgery", "emergency"]
      }
    },
    "additionalProperties": false,
    "required": ["patient_name", "appointment_type"]
  }
}

---

## Structured Outputs: Task Creation Schema (json)

URL: https://console.groq.com/docs/structured-outputs/scripts/task-creation-schema.json

{
  "name": "create_task",
  "description": "Creates a new task in the project management system",
  "strict": true,
  "parameters": {
    "type": "object",
    "properties": {
      "title": {
        "type": "string",
        "description": "The task title or summary"
      },
      "priority": {
        "type": "string",
        "description": "Task priority level",
        "enum": ["low", "medium", "high", "urgent"]
      }
    },
    "additionalProperties": false,
    "required": ["title", "priority"]
  }
}

---

## Structured Outputs: Support Ticket Zod.doc (ts)

URL: https://console.groq.com/docs/structured-outputs/scripts/support-ticket-zod.doc
```

Example 3 (unknown):
```unknown
---

## Structured Outputs: Email Classification Response (json)

URL: https://console.groq.com/docs/structured-outputs/scripts/email-classification-response.json
```

Example 4 (unknown):
```unknown
---

## Structured Outputs: Step2 Example (py)

URL: https://console.groq.com/docs/structured-outputs/scripts/step2-example.py

from groq import Groq
import json

client = Groq()

response = client.chat.completions.create(
    model="moonshotai/kimi-k2-instruct-0905",
    messages=[
        {"role": "system", "content": "You are a helpful math tutor. Guide the user through the solution step by step."},
        {"role": "user", "content": "how can I solve 8x + 7 = -23"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "math_response",
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "explanation": {"type": "string"},
                                "output": {"type": "string"}
                            },
                            "required": ["explanation", "output"],
                            "additionalProperties": False
                        }
                    },
                    "final_answer": {"type": "string"}
                },
                "required": ["steps", "final_answer"],
                "additionalProperties": False
            }
        }
    }
)

result = json.loads(response.choices[0].message.content)
print(json.dumps(result, indent=2))

---

## Structured Outputs: Api Response Validation (py)

URL: https://console.groq.com/docs/structured-outputs/scripts/api-response-validation.py
```

---

## Getting the base64 string

**URL:** llms-txt#getting-the-base64-string

**Contents:**
- Vision: Vision (py)
- Vision: Multiturn (py)
- Images and Vision

base64_image = encode_image(image_path)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)

print(chat_completion.choices[0].message.content)

## Vision: Vision (py)

URL: https://console.groq.com/docs/vision/scripts/vision.py

from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f2/LPU-v1-die.jpg"
                    }
                }
            ]
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
)

print(completion.choices[0].message)

## Vision: Multiturn (py)

URL: https://console.groq.com/docs/vision/scripts/multiturn.py

from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/d/da/SF_From_Marin_Highlands3.jpg"
                    }
                }
            ]
        },
        {
            "role": "user",
            "content": "Tell me more about the area."
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
)

print(completion.choices[0].message)

URL: https://console.groq.com/docs/vision

---

## Log the tools that were used to generate the response

**URL:** llms-txt#log-the-tools-that-were-used-to-generate-the-response

**Contents:**
- Compound: Fact Checker.doc (ts)
- Compound: Version (py)
- Example 1: Error Explanation (might trigger search)

print(response.choices[0].message.executed_tools)
javascript
import Groq from "groq-sdk";

const groq = new Groq();

export async function main() {
  const user_query = "What were the main highlights from the latest Apple keynote event?"
  // Or: "What's the current weather in San Francisco?"
  // Or: "Summarize the latest developments in fusion energy research this week."

const completion = await groq.chat.completions.create({
    messages: [
      {
        role: "user",
        content: user_query,
      },
    ],
    // The *only* change needed: Specify the compound model!
    model: "groq/compound",
  });

console.log(`Query: ${user_query}`);
  console.log(`Compound Response:\n${completion.choices[0]?.message?.content || ""}`);

// You might also inspect chat_completion.choices[0].message.executed_tools
  // if you want to see if/which tool was used, though it's not necessary.
}

main();
python
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

**Examples:**

Example 1 (unknown):
```unknown
---

## Compound: Fact Checker.doc (ts)

URL: https://console.groq.com/docs/compound/scripts/fact-checker.doc
```

Example 2 (unknown):
```unknown
---

## Compound: Version (py)

URL: https://console.groq.com/docs/compound/scripts/version.py

from groq import Groq

client = Groq(
    default_headers={
        "Groq-Model-Version": "latest"
    }
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "What is the weather today?",
        }
    ],
    model="groq/compound",
)

print(chat_completion.choices[0].message.content)

---

## Example 1: Error Explanation (might trigger search)

URL: https://console.groq.com/docs/compound/scripts/code-debugger.py
```

---

## Make the initial request

**URL:** llms-txt#make-the-initial-request

response = client.chat.completions.create(
    model=model, messages=messages, tools=tools, tool_choice="auto", max_completion_tokens=4096, temperature=0.5
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls

---

## print(completion.choices[0].message.executed_tools)

**URL:** llms-txt#print(completion.choices[0].message.executed_tools)

**Contents:**
- Compound: Natural Language (js)
- Compound: Executed Tools (js)
- Built In Tools: Enable Specific Tools (py)
- Built In Tools: Code Execution Only (py)
- Built In Tools: Code Execution Only (js)
- Built In Tools: Enable Specific Tools (js)
- Built-in Tools

javascript
import Groq from "groq-sdk";

const groq = new Groq();

export async function main() {
  // Example 1: Calculation
  const computationQuery = "Calculate the monthly payment for a $30,000 loan over 5 years at 6% annual interest.";

// Example 2: Simple code execution
  const codeQuery = "What is the output of this Python code snippet: `data = {'a': 1, 'b': 2}; print(data.keys())`";

// Choose one query to run
  const selectedQuery = computationQuery;

const completion = await groq.chat.completions.create({
    messages: [
      {
        role: "system",
        content: "You are a helpful assistant capable of performing calculations and executing simple code when asked.",
      },
      {
        role: "user",
        content: selectedQuery,
      }
    ],
    // Use the compound model
    model: "groq/compound-mini",
  });

console.log(`Query: ${selectedQuery}`);
  console.log(`Compound Mini Response:\n${completion.choices[0]?.message?.content || ""}`);
}

main();
javascript
import Groq from 'groq-sdk';

const groq = new Groq();

async function main() {
  const response = await groq.chat.completions.create({
    model: 'groq/compound',
    messages: [
      {
        role: 'user',
        content: 'What did Groq release last week?'
      }
    ]
  })
  // Log the tools that were used to generate the response
  console.log(response.choices[0].message.executed_tools)
}
main();
javascript
import Groq from "groq-sdk";

const groq = new Groq();

const response = await groq.chat.completions.create({
  model: "groq/compound",
  messages: [
    {
      role: "user",
      content: "Calculate the square root of 12345"
    }
  ],
  compound_custom: {
    tools: {
      enabled_tools: ["code_interpreter"]
    }
  }
});
javascript
import Groq from "groq-sdk";

const groq = new Groq({
  defaultHeaders: {
    "Groq-Model-Version": "latest"
  }
});

const response = await groq.chat.completions.create({
  model: "groq/compound",
  messages: [
    {
      role: "user",
      content: "Search for recent AI developments and then visit the Groq website"
    }
  ],
  compound_custom: {
    tools: {
      enabled_tools: ["web_search", "visit_website"]
    }
  }
});
```

URL: https://console.groq.com/docs/compound/built-in-tools

**Examples:**

Example 1 (unknown):
```unknown
---

## Compound: Natural Language (js)

URL: https://console.groq.com/docs/compound/scripts/natural-language
```

Example 2 (unknown):
```unknown
---

## Compound: Executed Tools (js)

URL: https://console.groq.com/docs/compound/scripts/executed_tools
```

Example 3 (unknown):
```unknown
---

## Built In Tools: Enable Specific Tools (py)

URL: https://console.groq.com/docs/compound/built-in-tools/scripts/enable-specific-tools.py

from groq import Groq

client = Groq(
    default_headers={
        "Groq-Model-Version": "latest"
    }
)

response = client.chat.completions.create(
    model="groq/compound",
    messages=[
        {
            "role": "user",
            "content": "Search for recent AI developments and then visit the Groq website"
        }
    ],
    compound_custom={
        "tools": {
            "enabled_tools": ["web_search", "visit_website"]
        }
    }
)

---

## Built In Tools: Code Execution Only (py)

URL: https://console.groq.com/docs/compound/built-in-tools/scripts/code-execution-only.py

from groq import Groq

client = Groq()

response = client.chat.completions.create(
    model="groq/compound",
    messages=[
        {
            "role": "user", 
            "content": "Calculate the square root of 12345"
        }
    ],
    compound_custom={
        "tools": {
            "enabled_tools": ["code_interpreter"]
        }
    }
)

---

## Built In Tools: Code Execution Only (js)

URL: https://console.groq.com/docs/compound/built-in-tools/scripts/code-execution-only
```

Example 4 (unknown):
```unknown
---

## Built In Tools: Enable Specific Tools (js)

URL: https://console.groq.com/docs/compound/built-in-tools/scripts/enable-specific-tools
```

---

## Print the completion returned by the LLM.

**URL:** llms-txt#print-the-completion-returned-by-the-llm.

**Contents:**
- Text Generation

print(chat_completion.choices[0].message.content)
```

URL: https://console.groq.com/docs/text-chat

---

## Print the incremental deltas returned by the LLM.

**URL:** llms-txt#print-the-incremental-deltas-returned-by-the-llm.

**Contents:**
- Required parameters
- Text Chat: System Prompt (py)
- Text Chat: Prompt Engineering (js)
- Required parameters

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
python
import asyncio

from groq import AsyncGroq

async def main():
    client = AsyncGroq()

chat_completion = await client.chat.completions.create(
        #
        # Required parameters
        #
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": "Explain the importance of fast language models",
            }
        ],

# The language model which will generate the completion.
        model="llama-3.3-70b-versatile",

#
        # Optional parameters
        #

# Controls randomness: lowering results in less random completions.
        # As the temperature approaches zero, the model will become
        # deterministic and repetitive.
        temperature=0.5,

# The maximum number of tokens to generate. Requests can use up to
        # 2048 tokens shared between prompt and completion.
        max_completion_tokens=1024,

# Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        top_p=1,

# A stop sequence is a predefined or user-specified text string that
        # signals an AI to stop generating content, ensuring its responses
        # remain focused and concise. Examples include punctuation marks and
        # markers like "[end]".
        stop=None,

# If set, partial message deltas will be sent.
        stream=False,
    )

# Print the completion returned by the LLM.
    print(chat_completion.choices[0].message.content)

asyncio.run(main())
python
from groq import Groq

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": "You are a data analysis API that performs sentiment analysis on text. Respond only with JSON using this format: {\"sentiment_analysis\": {\"sentiment\": \"positive|negative|neutral\", \"confidence_score\": 0.95, \"key_phrases\": [{\"phrase\": \"detected key phrase\", \"sentiment\": \"positive|negative|neutral\"}], \"summary\": \"One sentence summary of the overall sentiment\"}}"
        },
        {
            "role": "user",
            "content": "Analyze the sentiment of this customer review: 'I absolutely love this product! The quality exceeded my expectations, though shipping took longer than expected.'"
        }
    ],
    response_format={"type": "json_object"}
)

print(response.choices[0].message.content)
javascript
import { Groq } from "groq-sdk";

const client = new Groq();

// Example of a poorly designed prompt
const poorPrompt = `
Give me information about a movie in JSON format.
`;

// Example of a well-designed prompt
const effectivePrompt = `
You are a movie database API. Return information about a movie with the following 
JSON structure:

{
  "title": "string",
  "year": number,
  "director": "string",
  "genre": ["string"],
  "runtime_minutes": number,
  "rating": number (1-10 scale),
  "box_office_millions": number,
  "cast": [
    {
      "actor": "string",
      "character": "string"
    }
  ]
}

The response must:
1. Include ALL fields shown above
2. Use only the exact field names shown
3. Follow the exact data types specified
4. Contain ONLY the JSON object and nothing else

IMPORTANT: Do not include any explanatory text, markdown formatting, or code blocks.
`;

// Function to run the completion and display results
async function getMovieData(prompt, title = "Example") {
  console.log(`\n--- ${title} ---`);
  
  try {
    const completion = await client.chat.completions.create({
      model: "openai/gpt-oss-20b",
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: prompt },
        { role: "user", content: "Tell me about The Matrix" },
      ],
    });
    
    const responseContent = completion.choices[0].message.content;
    console.log("Raw response:");
    console.log(responseContent);
    
    // Try to parse as JSON
    try {
      const movieData = JSON.parse(responseContent || "");
      console.log("\nSuccessfully parsed as JSON!");
      
      // Check for expected fields
      const expectedFields = ["title", "year", "director", "genre", 
                            "runtime_minutes", "rating", "box_office_millions", "cast"];
      const missingFields = expectedFields.filter(field => !(field in movieData));
      
      if (missingFields.length > 0) {
        console.log(`Missing fields: ${missingFields.join(', ')}`);
      } else {
        console.log("All expected fields present!");
      }
      
      return movieData;
    } catch (syntaxError) {
      console.log("\nFailed to parse as JSON. Response is not valid JSON.");
      return null;
    }
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

// Compare the results of both prompts
async function comparePrompts() {
  await getMovieData(poorPrompt, "Poor Prompt Example");
  await getMovieData(effectivePrompt, "Effective Prompt Example");
}

// Run the examples
comparePrompts();
python
from groq import Groq

chat_completion = client.chat.completions.create(
    #
    # Required parameters
    #
    messages=[
        # Set an optional system message. This sets the behavior of the
        # assistant and can be used to provide specific instructions for
        # how it should behave throughout the conversation.
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        # Set a user message for the assistant to respond to.
        {
            "role": "user",
            "content": "Count to 10.  Your response must begin with \"1, \".  example: 1, 2, 3, ...",
        }
    ],

# The language model which will generate the completion.
    model="llama-3.3-70b-versatile",

#
    # Optional parameters
    #

# Controls randomness: lowering results in less random completions.
    # As the temperature approaches zero, the model will become deterministic
    # and repetitive.
    temperature=0.5,

# The maximum number of tokens to generate. Requests can use up to
    # 2048 tokens shared between prompt and completion.
    max_completion_tokens=1024,

# Controls diversity via nucleus sampling: 0.5 means half of all
    # likelihood-weighted options are considered.
    top_p=1,

# A stop sequence is a predefined or user-specified text string that
    # signals an AI to stop generating content, ensuring its responses
    # remain focused and concise. Examples include punctuation marks and
    # markers like "[end]".
    # For this example, we will use ", 6" so that the llm stops counting at 5.
    # If multiple stop values are needed, an array of string may be passed,
    # stop=[", 6", ", six", ", Six"]
    stop=", 6",

# If set, partial message deltas will be sent.
    stream=False,
)

**Examples:**

Example 1 (unknown):
```unknown
---

## Required parameters

URL: https://console.groq.com/docs/text-chat/scripts/performing-async-chat-completion.py
```

Example 2 (unknown):
```unknown
---

## Text Chat: System Prompt (py)

URL: https://console.groq.com/docs/text-chat/scripts/system-prompt.py
```

Example 3 (unknown):
```unknown
---

## Text Chat: Prompt Engineering (js)

URL: https://console.groq.com/docs/text-chat/scripts/prompt-engineering
```

Example 4 (unknown):
```unknown
---

## Required parameters

URL: https://console.groq.com/docs/text-chat/scripts/streaming-chat-completion-with-stop.py
```

---

## Processing Tier Selection Logic

**URL:** llms-txt#processing-tier-selection-logic

**Contents:**
  - Batch Processing
- Streaming Implementation
  - Server-Sent Events Best Practices

if real_time_required and throughput_need != "high":
    return "on_demand"  
elif throughput_need == "high" and cost_priority != "critical":
    return "flex"  
elif real_time_required and throughput_need == "variable":
    return "auto"  
elif cost_priority == "critical":
    return "batch"  
else:
    return "on_demand" 
python
import os
from groq import Groq

def stream_response(prompt):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    stream = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

**Examples:**

Example 1 (unknown):
```unknown
### Batch Processing

[Batch Processing](/docs/batch) enables cost-effective asynchronous processing with a completion window, optimized for scenarios where immediate responses aren't required.
<br/> 
**Batch API Overview**: The Groq Batch API processes large-scale workloads asynchronously, offering significant advantages for high-volume use cases:

- **Higher rate limits**: Process thousands of requests per batch with no impact on standard API rate limits
- **Cost efficiency**: 50% cost discount compared to synchronous APIs
- **Flexible processing windows**: 24-hour to 7-day completion timeframes based on workload requirements
- **Rate limit isolation**: Batch processing doesn't consume your standard API quotas

<br/>
**Latency Considerations**: While batch processing trades immediate response for efficiency, understanding its latency characteristics helps optimize workload planning:

- **Submission latency**: Minimal overhead for batch job creation and validation
- **Queue processing**: Variable based on system load and batch size
- **Completion notification**: Webhook or polling-based status updates
- **Result retrieval**: Standard API latency for downloading completed outputs

<br/>
**Optimal Use Cases**: Batch processing excels for workloads where processing time flexibility enables significant cost and throughput benefits: large dataset analysis, content generation pipelines, model evaluation suites, and scheduled data enrichment tasks.

## Streaming Implementation

### Server-Sent Events Best Practices

Implement streaming to improve perceived latency:
<br/>

**Streaming Implementation**:
```

---

## Start the conversation

**URL:** llms-txt#start-the-conversation

**Contents:**
- Content Moderation: Llamaguard Chat Completion (json)
- Content Moderation: Llamaguard Chat Completion (py)
- Content Moderation: Llamaguard Chat Completion (js)
- Content Moderation

user_proxy.initiate_chat(
    assistant,
    message="""Let's do two things:
    1. Get the weather for Berlin, Istanbul, and San Francisco
    2. Write a Python script to create a bar chart comparing their temperatures"""
)
python
import os

from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
        "role": "user",
        "content": "Write a script that can be used to hack into a Wi-Fi network"
        }
    ],
    model="meta-llama/Llama-Guard-4-12B",
)

print(chat_completion.choices[0].message.content)
javascript
import Groq from "groq-sdk";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const result = await groq.chat.completions.create({
    messages: [
        {
            "role": "user",
            "content": "Write a script that can be used to hack into a Wi-Fi network"
        }
    ],
    model: "meta-llama/Llama-Guard-4-12B",
});

console.log(result.choices[0]?.message?.content);
```

## Content Moderation

URL: https://console.groq.com/docs/content-moderation

**Examples:**

Example 1 (unknown):
```unknown
**Challenge:** Add to the above example and create a multi-agent [`GroupChat`](https://microsoft.github.io/autogen/0.2/docs/topics/groupchat/customized_speaker_selection) workflow!


For more detailed documentation and resources on building agentic applications with Groq and AutoGen, see:
- [AutoGen Documentation](https://microsoft.github.io/autogen/0.2/docs/topics/non-openai-models/cloud-groq/)
- [AutoGroq](https://github.com/jgravelle/AutoGroq)

---

## Content Moderation: Llamaguard Chat Completion (json)

URL: https://console.groq.com/docs/content-moderation/scripts/llamaguard-chat-completion.json

{
  "messages": [
    {
      "role": "user",
      "content": "Write a script that can be used to hack into a Wi-Fi network"
    }
  ],
  "model": "meta-llama/Llama-Guard-4-12B"
}

---

## Content Moderation: Llamaguard Chat Completion (py)

URL: https://console.groq.com/docs/content-moderation/scripts/llamaguard-chat-completion.py
```

Example 2 (unknown):
```unknown
---

## Content Moderation: Llamaguard Chat Completion (js)

URL: https://console.groq.com/docs/content-moderation/scripts/llamaguard-chat-completion
```

---

## The API will stop generation when '###' is encountered and will NOT include '###' in the response.

**URL:** llms-txt#the-api-will-stop-generation-when-'###'-is-encountered-and-will-not-include-'###'-in-the-response.

**Contents:**
- Some creativity allowed
- Prompting: Stop (js)
- Prompt Basics

from groq import Groq

client = Groq()
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Provide a 2-sentence summary of the concept of 'artificial general intelligence'. End your summary with '###'."
        }
        # Model's goal before stop sequence removal might be:
        # "Artificial general intelligence (AGI) refers to a type of AI that possesses the ability to understand, learn, and apply knowledge across a wide range of tasks at a level comparable to that of a human being. This contrasts with narrow AI, which is designed for specific tasks. ###"
    ],
    model="llama-3.1-8b-instant",
    stop=["###"],
    max_tokens=100 # Ensure enough tokens for the summary + stop sequence
)

print(chat_completion.choices[0].message.content)

## Some creativity allowed

URL: https://console.groq.com/docs/prompting/scripts/seed.py

## Prompting: Stop (js)

URL: https://console.groq.com/docs/prompting/scripts/stop

URL: https://console.groq.com/docs/prompting

**Examples:**

Example 1 (python):
```python
from groq import Groq

client = Groq()
chat_completion = client.chat.completions.create(
    messages=[
      { "role": "system", "content": "You are a creative storyteller." },
      { "role": "user", "content": "Write a brief opening line to a mystery novel." }
    ],
    model="llama-3.1-8b-instant",
    temperature=0.8,  # Some creativity allowed
    seed=700,  # Deterministic seed
    max_tokens=100
)

print(chat_completion.choices[0].message.content)
```

Example 2 (javascript):
```javascript
// Using a custom stop sequence for structured, concise output.
// The model is instructed to produce '###' at the end of the desired content.
// The API will stop generation when '###' is encountered and will NOT include '###' in the response.

import { Groq } from "groq-sdk"

const groq = new Groq()
const response = await groq.chat.completions.create({
  messages: [
    {
      role: "user",
      content: "Provide a 2-sentence summary of the concept of 'artificial general intelligence'. End your summary with '###'."
    }
    // Model's goal before stop sequence removal might be:
    // "Artificial general intelligence (AGI) refers to a type of AI that possesses the ability to understand, learn, and apply knowledge across a wide range of tasks at a level comparable to that of a human being. This contrasts with narrow AI, which is designed for specific tasks. ###"
  ],
  model: "llama-3.1-8b-instant",
  stop: ["###"],
  max_tokens: 100 // Ensure enough tokens for the summary + stop sequence
});

console.log(response.choices[0].message.content)
```

---

## Use the create method to create new message

**URL:** llms-txt#use-the-create-method-to-create-new-message

**Contents:**
  - 5. Visualize model usage on the MLflow tracing dashboard:
- Additional Resources
- How Groq Uses Your Feedback
- How Groq Uses Your Feedback
- What We Collect
- How Feedback Is Reviewed
- How We Use Your Feedback
- Retention
- Learn More
- LoRA Inference on Groq

message = client.chat.completions.create(
    model="qwen-2.5-32b",
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs.",
        }
    ],
)

print(message.choices[0].message.content)
```

### 5. Visualize model usage on the MLflow tracing dashboard:

Now traces for your Groq usage are captured by MLflow! Let's get insights into our application's activities by visiting the MLflow tracking server
we set in Step 4 above (`mlflow.set_tracking_uri("http://localhost:5000")`), which we can do by opening http://localhost:5000 in our browser.

![mlflow tracing dashboard](/mlflow.png)

## Additional Resources
For more configuration and detailed resources for managing your Groq applications with MLflow, see:
- [Getting Started with MLflow](https://mlflow.org/docs/latest/getting-started/index.html)
- [MLflow LLMs Overview](https://mlflow.org/docs/latest/llms/index.html)
- [MLflow Tracing for LLM Observability](https://mlflow.org/docs/latest/llms/tracing/index.html)

## How Groq Uses Your Feedback

URL: https://console.groq.com/docs/feedback-policy

## How Groq Uses Your Feedback

Your feedback is essential to making GroqCloud and our products safer, more reliable, and more useful. This page explains how we collect, review, and retain feedback in accordance with [Groq's Privacy Policy](https://groq.com/privacy-policy).

When you submit feedback—whether through the in‑product **“Provide Feedback”** button, a survey, or a support ticket—we may receive:

- **Your written comments or attachments** (e.g. screenshots, logs, or files you choose to include).
- **Conversation context**, such as prompts (Inputs) and AI responses (Outputs) related to the feedback.
- **Metadata** like time stamps and product versions that help us reproduce the issue.

We do not use this feedback mechanisms to collect any personal information such as passwords, payment details, or other sensitive personal data, and we ask that you avoid sharing such information in feedback.

## How Feedback Is Reviewed

- Groq's trust & safety, customer and technical support teams manually review a subset of feedback to pinpoint issues, bugs, or UX friction that automated systems can miss.

## How We Use Your Feedback

Your feedback is processed **consistent with the [Groq Privacy Policy](https://groq.com/privacy-policy)** and serves two primary purposes:

- **Improve product quality** – reproducing bugs, refining model outputs, and enhancing documentation.
- **Keep our systems safe** – patterns in reports help us detect and block unsafe content or behavior.

Reviewed feedback, conversation snippets, and related metadata are **stored for up to 3 years.** After that period, the data is permanently deleted. You can ask us to delete your account and corresponding personal information at any time.

See the [Groq Privacy Policy](https://groq.com/privacy-policy).

## LoRA Inference on Groq

URL: https://console.groq.com/docs/lora

---

## You might also inspect chat_completion.choices[0].message.executed_tools

**URL:** llms-txt#you-might-also-inspect-chat_completion.choices[0].message.executed_tools

---
