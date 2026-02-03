# Mistral - Agents

**Pages:** 5

---

## Create your agents

**URL:** llms-txt#create-your-agents

**Contents:**
  - Define Handoffs Responsibilities

finance_agent = client.beta.agents.create(
    model="mistral-large-latest",
    description="Agent used to answer financial related requests",
    name="finance-agent",
)
web_search_agent = client.beta.agents.create(
    model="mistral-large-latest",
    description="Agent that can search online for any information if needed",
    name="websearch-agent",
    tools=[{"type": "web_search"}],
)
ecb_interest_rate_agent = client.beta.agents.create(
    model="mistral-large-latest",
    description="Can find the current interest rate of the European central bank",
    name="ecb-interest-rate-agent",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_european_central_bank_interest_rate",
                "description": "Retrieve the real interest rate of European central bank.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "date",
                    ]
                },
            },
        },
    ],
)
graph_agent = client.beta.agents.create(
    model="mistral-large-latest",
    name="graph-drawing-agent",
    description="Agent used to create graphs using the code interpreter tool.",
    instructions="Use the code interpreter tool when you have to draw a graph.",
    tools=[{"type": "code_interpreter"}]
)
calculator_agent = client.beta.agents.create(
    model="mistral-large-latest",
    name="calculator-agent",
    description="Agent used to make detailed calculations",
    instructions="When doing calculations explain step by step what you are doing.",
    completion_args=CompletionArgs(
          response_format=ResponseFormat(
            type="json_schema",
            json_schema=JSONSchema(
                name="calc_result",
                schema=CalcResult.model_json_schema(),
            )
        )
    )
)
typescript

typescript
const CalcResult = z.object({
        reasoning: z.string(),
        result: z.string(),
    });

let financeAgent = await client.beta.agents.create({
    model: "mistral-large-latest",
    description: "Agent used to answer financial related requests",
    name: "finance-agent",
});

let webSearchAgent = await client.beta.agents.create({
    model: "mistral-large-latest",
    description: "Agent that can search online for any information if needed",
    name: "websearch-agent",
    tools: [{ type: "web_search" }],
});

let ecbInterestRateAgent = await client.beta.agents.create({
    model: "mistral-large-latest",
    description: "Can find the current interest rate of the European central bank",
    name: "ecb-interest-rate-agent",
    tools: [
        {
            type: "function",
            function: {
                name: "getEuropeanCentralBankInterestRate",
                description: "Retrieve the real interest rate of European central bank.",
                parameters: {
                    type: "object",
                    properties: {
                        date: {
                            type: "string",
                        },
                    },
                    required: ["date"],
                },
            },
        },
    ],
});

const graphAgent = await client.beta.agents.create({
    model: "mistral-large-latest",
    name: "graph-drawing-agent",
    description: "Agent used to create graphs using the code interpreter tool.",
    instructions: "Use the code interpreter tool when you have to draw a graph.",
    tools: [{ type: "code_interpreter" }],
});

const calculatorAgent = await client.beta.agents.create({
    model: "mistral-large-latest",
    name: "calculator-agent",
    description: "Agent used to make detailed calculations",
    instructions: "When doing calculations explain step by step what you are doing.",
    completionArgs: {
        responseFormat: responseFormatFromZodObject(CalcResult)
    },
});
bash
curl --location "https://api.mistral.ai/v1/agents" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
     "model": "mistral-large-latest",
     "name": "finance-agent",
     "description": "Agent used to answer financial related requests"
  }'
py

**Examples:**

Example 1 (unknown):
```unknown
</TabItem>

  <TabItem value="typescript" label="typescript">

First, let's make the following import:
```

Example 2 (unknown):
```unknown
Then, we define and create our agents:
```

Example 3 (unknown):
```unknown
</TabItem>

  <TabItem value="curl" label="curl">
```

Example 4 (unknown):
```unknown
</TabItem>
</Tabs>

### Define Handoffs Responsibilities

<div style={{ textAlign: 'center' }}>
  <img
    src="/img/responsibilities_handoffs.png"
    alt="handoffs_graph"
    width="800"
    style={{ borderRadius: '15px' }}
  />
</div>

Once all our Agents created, we update our previous defined Agents with a list of `handoffs` available.
<Tabs groupId="code">
  <TabItem value="python" label="python" default>
```

---

## List agent entities.

**URL:** llms-txt#list-agent-entities.

Source: https://docs.mistral.ai/api/#tag/agents_api_v1_agents_list

---

## Retrieve an agent entity.

**URL:** llms-txt#retrieve-an-agent-entity.

Source: https://docs.mistral.ai/api/#tag/agents_api_v1_agents_get

get /v1/agents/{agent_id}

---

## Update an agent entity.

**URL:** llms-txt#update-an-agent-entity.

Source: https://docs.mistral.ai/api/#tag/agents_api_v1_agents_update

patch /v1/agents/{agent_id}

---

## Update an agent version.

**URL:** llms-txt#update-an-agent-version.

Source: https://docs.mistral.ai/api/#tag/agents_api_v1_agents_update_version

patch /v1/agents/{agent_id}/version

---
