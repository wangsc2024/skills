# Mistral - Chat Completion

**Pages:** 25

---

## Agents Completion

**URL:** llms-txt#agents-completion

Source: https://docs.mistral.ai/api/#tag/agents_completion_v1_agents_completions_post

post /v1/agents/completions

---

## Allow the ecb_interest_rate_agent to handoff the conversation to the graph_agent or calculator_agent

**URL:** llms-txt#allow-the-ecb_interest_rate_agent-to-handoff-the-conversation-to-the-graph_agent-or-calculator_agent

ecb_interest_rate_agent = client.beta.agents.update(
    agent_id=ecb_interest_rate_agent.id,
    handoffs=[graph_agent.id, calculator_agent.id]
)

---

## Allow the finance_agent to handoff the conversation to the ecb_interest_rate_agent or web_search_agent

**URL:** llms-txt#allow-the-finance_agent-to-handoff-the-conversation-to-the-ecb_interest_rate_agent-or-web_search_agent

finance_agent = client.beta.agents.update(
    agent_id=finance_agent.id,
    handoffs=[ecb_interest_rate_agent.id, web_search_agent.id]
)

---

## Allow the web_search_agent to handoff the conversation to the graph_agent or calculator_agent

**URL:** llms-txt#allow-the-web_search_agent-to-handoff-the-conversation-to-the-graph_agent-or-calculator_agent

**Contents:**
- How It Works
  - Example A
- Event type: agent.handoff
- Event type: tool.execution
- Event type: message.output
- Event type: agent.handoff
- Event type: message.output
  - Example B
- Event type: agent.handoff
- Event type: function.call

web_search_agent = client.beta.agents.update(
    agent_id=web_search_agent.id,
    handoffs=[graph_agent.id, calculator_agent.id]
)
typescript
// Allow the financeAgent to handoff the conversation to the ecbInterestRateAgent or webSearchAgent
financeAgent = await client.beta.agents.update({
    agentId: financeAgent.id,
    agentUpdateRequest:{
        handoffs: [ecbInterestRateAgent.id, webSearchAgent.id]
    }
});

// Allow the ecbInterestRateAgent to handoff the conversation to the grapAgent or calculatorAgent
ecbInterestRateAgent = await client.beta.agents.update({
    agentId: ecbInterestRateAgent.id,
    agentUpdateRequest:{
        handoffs: [graphAgent.id, calculatorAgent.id]
    }
});

// Allow the webSearchAgent to handoff the conversation to the graphAgent or calculatorAgent
webSearchAgent = await client.beta.agents.update({
    agentId: webSearchAgent.id,
    agentUpdateRequest:{
        handoffs: [graphAgent.id, calculatorAgent.id]
    }
});
bash
curl --location "https://api.mistral.ai/v1/agents/<web_search_id>" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
     "handoffs": ["<graph_agent_id>", "<calculator_agent_id>"]
  }'
py
response = client.beta.conversations.start(
    agent_id=finance_agent.id,
    inputs="Fetch the current US bank interest rate and calculate the compounded effect if investing for the next 10y"
)
typescript
let response = await client.beta.conversations.start({
    agentId:financeAgent.id,
    inputs:"Fetch the current US bank interest rate and calculate the compounded effect if investing for the next 10y"
});
shell
Conversation started: conv_067f7fce0aba70728000b32dcb0ac7e5

## Event type: agent.handoff

agent_id='ag_067f7fce04517b60800013b213ff2acb' agent_name='websearch-agent' object='conversation.entry' type='agent.handoff' created_at=datetime.datetime(2025, 4, 10, 17, 16, 18, 952817, tzinfo=TzInfo(UTC)) id='handoff_067f7fce2f3f7423800094104f3e3589'

## Event type: tool.execution

name='web_search' object='conversation.entry' type='tool.execution' created_at=datetime.datetime(2025, 4, 10, 17, 16, 23, 12996, tzinfo=TzInfo(UTC)) id='tool_exec_067f7fce7035747e800085153507b345'

## Event type: message.output

content=[TextChunk(text='The current US bank interest rate is 4.50 percent', type='text'), ToolReferenceChunk(tool='web_search', title='United States Fed Funds Interest Rate', type='tool_reference', url='https://tradingeconomics.com/united-states/interest-rate'), TextChunk(text='.\n\nI will now handoff the conversation to the calculator agent to calculate the compounded effect if investing for the next 10 years.', type='text')] object='conversation.entry' type='message.output' created_at=datetime.datetime(2025, 4, 10, 17, 16, 23, 14612, tzinfo=TzInfo(UTC)) id='msg_067f7fce703b7e01800045b2309a0750' agent_id='ag_067f7fce04517b60800013b213ff2acb' model='mistral-medium-2505' role='assistant'

## Event type: agent.handoff

agent_id='ag_067f7fce017f71a580001bf69f2cc11e' agent_name='calculator-agent' object='conversation.entry' type='agent.handoff' created_at=datetime.datetime(2025, 4, 10, 17, 16, 23, 14726, tzinfo=TzInfo(UTC)) id='handoff_067f7fce703c753680006aedb42ba7b7'

## Event type: message.output

content=' {"result": "The future value of the investment after 10 years is $1,540.10.", "reasoning": "To calculate the compounded effect of investing at the current US bank interest rate of 4.50% for the next 10 years, we use the formula for compound interest: A = P(1 + r/n)^(nt), where A is the amount of money accumulated after n years, including interest. P is the principal amount (the initial amount of money). r is the annual interest rate (decimal). n is the number of times that interest is compounded per year. t is the time the money is invested for, in years. Assuming an initial investment (P) of $1,000, an annual interest rate (r) of 4.50% (or 0.045 as a decimal), compounded annually (n = 1), over 10 years (t = 10): A = 1000(1 + 0.045/1)^(1*10) = 1000(1 + 0.045)^10 = 1000(1.045)^10 ≈ 1540.10. Therefore, the future value of the investment after 10 years is approximately $1,540.10."}' object='conversation.entry' type='message.output' created_at=datetime.datetime(2025, 4, 10, 17, 16, 30, 145207, tzinfo=TzInfo(UTC)) id='msg_067f7fcee2527cf08000744d983639dc' agent_id='ag_067f7fce017f71a580001bf69f2cc11e' model='mistral-medium-2505' role='assistant'
py
from mistralai import FunctionResultEntry

response = client.beta.conversations.start(
    agent_id=finance_agent.id,
    inputs="Given the interest rate of the European Central Bank as of jan 2025, plot a graph of the compounded interest rate over the next 10 years"
)
if response.outputs[-1].type == "function.call" and response.outputs[-1].name == "get_european_central_bank_interest_rate":

# Add a dummy result for the function call
    user_entry = FunctionResultEntry(
        tool_call_id=response.outputs[-1].tool_call_id,
        result="2.5%",
    )
    response = client.beta.conversations.append(
        conversation_id=response.conversation_id,
        inputs=[user_entry]
    )
py
from mistralai.models import ToolFileChunk

for i, chunk in enumerate(response.outputs[-1].content):
    # Check if chunk corresponds to a ToolFileChunk
    if isinstance(chunk, ToolFileChunk):

# Download using the ToolFileChunk ID
      file_bytes = client.files.download(file_id=chunk.file_id).read()

# Save the file locally
      with open(f"plot_generated_{i}.png", "wb") as file:
          file.write(file_bytes)
typescript

typescript
response = await client.beta.conversations.start({
    agentId:financeAgent.id,
    inputs:"Given the interest rate of the European Central Bank as of jan 2025, plot a graph of the compounded interest rate over the next 10 years"
});

let output = response.outputs[response.outputs.length - 1];

if (output.type === "function.call" && output.name === "getEuropeanCentralBankInterestRate") {
    // Add a dummy result for the function call
    let userEntry: FunctionResultEntry = {
        toolCallId: output.toolCallId,
        result: "2.5%",
    };

response = await client.beta.conversations.append({
        conversationId:response.conversationId,
        conversationAppendRequest:{
            inputs:[userEntry]
        }
    });
}
bash
curl --location "https://api.mistral.ai/v1/files/<file_id>/content" \
     --header 'Accept: application/octet-stream' \
     --header 'Accept-Encoding: gzip, deflate, zstd' \
     --header "Authorization: Bearer $MISTRAL_API_KEY"
shell
Conversation started: conv_067f7e71523d7be3800005c4ac560a7b

## Event type: agent.handoff

agent_id='ag_067f7e714f6e751480002beb3bfe0779' agent_name='ecb-interest-rate-agent' object='conversation.entry' type='agent.handoff' created_at=datetime.datetime(2025, 4, 10, 15, 43, 18, 590169, tzinfo=TzInfo(UTC)) id='handoff_067f7e71697176098000aa403030a74e'

## Event type: function.call

tool_call_id='NqCFiwvSV' name='get_european_central_bank_interest_rate' arguments='{"date": "2025-01-01"}' object='conversation.entry' type='function.call' created_at=datetime.datetime(2025, 4, 10, 15, 43, 20, 173505, tzinfo=TzInfo(UTC)) id='fc_067f7e7182c67b9c80006f27131026a8'

## User added event function.result:

tool_call_id='NqCFiwvSV' result='2.5%' object='conversation.entry' type='function.result' created_at=None id=None

## Event type: agent.handoff:

agent_id='ag_067f7e7147e077a280005b4ae524d317' agent_name='graph-drawing-agent' object='conversation.entry' type='agent.handoff' created_at=datetime.datetime(2025, 4, 10, 15, 43, 26, 261436, tzinfo=TzInfo(UTC)) id='handoff_067f7e71e42e7e2080009fc4fd68164a'

## Event type: message.output:

content="To plot the graph of the compounded interest rate over the next 10 years, we can use the formula for compound interest:\n\n\\[ A = P \\left(1 + \\frac{r}{n}\\right)^{nt} \\]\n\nwhere:\n- \\( A \\) is the amount of money accumulated after n years, including interest.\n- \\( P \\) is the principal amount (the initial amount of money).\n- \\( r \\) is the annual interest rate (decimal).\n- \\( n \\) is the number of times that interest is compounded per year.\n- \\( t \\) is the time the money is invested for, in years.\n\nGiven:\n- The annual interest rate \\( r = 2.5\\% = 0.025 \\).\n- Assuming the interest is compounded annually (\\( n = 1 \\)).\n- We will calculate the compounded amount for each year over the next 10 years.\n\nLet's assume the principal amount \\( P = 1000 \\) (you can choose any amount as it will not affect the rate plot).\n\nWe will calculate the compounded amount for each year and plot it." object='conversation.entry' type='message.output' created_at=datetime.datetime(2025, 4, 10, 15, 43, 39, 385339, tzinfo=TzInfo(UTC)) id='msg_067f7e72b62a768f800022b2504adfc9' agent_id='ag_067f7e7147e077a280005b4ae524d317' model='mistral-medium-2505' role='assistant'

## Event type: tool.execution:

name='code_interpreter' object='conversation.entry' type='tool.execution' created_at=datetime.datetime(2025, 4, 10, 15, 43, 39, 385463, tzinfo=TzInfo(UTC)) id='tool_exec_067f7e72b62a7e3a800072733a6a57f2'

## Event type: message.output:

content=[ToolFileChunk(tool='code_interpreter', file_id='40420c94-5f99-477f-8891-943f0defbe3b', type='tool_file', file_name='plot_0.png', file_type='png'), TextChunk(text='![Image](__emitted_0.png)\n\nThe graph shows the compounded interest over 10 years with an annual interest rate of 2.5%. The principal amount is set to $1000, and the interest is compounded once per year. The y-axis represents the amount of money, and the x-axis represents the number of years.', type='text')] object='conversation.entry' type='message.output' created_at=datetime.datetime(2025, 4, 10, 15, 43, 39, 898738, tzinfo=TzInfo(UTC)) id='msg_067f7e72be6173f48000e85e9976305a' agent_id='ag_067f7e7147e077a280005b4ae524d317' model='mistral-medium-2505' role='assistant'
python
#!/usr/bin/env python

from mistralai import Mistral
from mistralai.extra.run.context import RunContext
from mcp import StdioServerParameters
from mistralai.extra.mcp.stdio import MCPClientSTDIO
from pathlib import Path

from mistralai.types import BaseModel

**Examples:**

Example 1 (unknown):
```unknown
</TabItem>

  <TabItem value="typescript" label="typescript">
```

Example 2 (unknown):
```unknown
</TabItem>

  <TabItem value="curl" label="curl">
```

Example 3 (unknown):
```unknown
</TabItem>
</Tabs>

## How It Works

Our workflow and behavior are defined, now we can run it.

We created 5 agents, some of them have access to built-in tools, and others to local tools like `get_european_central_bank_interest_rate`.

It is now possible to have a chain of actions by sending a request to the `finance_agent`.

We also provide the parameter `handoff_execution`, which currently has two modes: `server` or `client`.
- `server`: Runs the handoff as expected internally on our cloud servers; this is the default setting.
- `client`: When a handoff is triggered, a response is provided directly to the user, enabling them to handle the handoff with control.

Let’s trigger two different behaviors as examples:

### Example A

**"Fetch the current US bank interest rate and calculate the compounded effect if investing for the next 10y"**

The first example asks for the US central bank interest rate, so we expect to involve the `websearch-agent` and then to calculate the compounded interest over 10 years. This should use the `calculator-agent` to do this.

<div style={{ textAlign: 'center' }}>
  <img
    src="/img/examplea_handoffs.png"
    alt="handoffs_graph_examplea"
    width="800"
    style={{ borderRadius: '15px' }}
  />
</div>

<Tabs groupId="code">
  <TabItem value="python" label="python" default>
```

Example 4 (unknown):
```unknown
</TabItem>

  <TabItem value="typescript" label="typescript">
```

---

## Append new entries to an existing conversation.

**URL:** llms-txt#append-new-entries-to-an-existing-conversation.

Source: https://docs.mistral.ai/api/#tag/agents_api_v1_conversations_append_stream

post /v1/conversations/{conversation_id}#stream

---

## Append the tool call message to the chat_history

**URL:** llms-txt#append-the-tool-call-message-to-the-chat_history

**Contents:**
  - Make the Final Chat Request
  - Extract and Print References

chat_history.append(tool_call_result)
python
chat_response = client.chat.complete(
    model=model,
    messages=chat_history,
    tools=[get_information_tool],
)

print(chat_response.choices[0].message.content)

[TextChunk(text='The Nobel Peace Prize for 2024 was awarded to the Japan Confederation of A- and H-Bomb Sufferers Organizations (Nihon Hidankyo) for their activism against nuclear weapons, including efforts by survivors of the atomic bombings of Hiroshima and Nagasaki', type='text'), ReferenceChunk(reference_ids=[0], type='reference'), TextChunk(text='.', type='text')]
python
from mistralai.models import TextChunk, ReferenceChunk

**Examples:**

Example 1 (unknown):
```unknown
### Make the Final Chat Request
```

Example 2 (unknown):
```unknown
Output:
```

Example 3 (unknown):
```unknown
### Extract and Print References
```

---

## Chat Classifications

**URL:** llms-txt#chat-classifications

Source: https://docs.mistral.ai/api/#tag/chat_classifications_v1_chat_classifications_post

post /v1/chat/classifications

---

## Chat Completion

**URL:** llms-txt#chat-completion

Source: https://docs.mistral.ai/api/#tag/chat_completion_v1_chat_completions_post

post /v1/chat/completions

---

## Chat Moderations

**URL:** llms-txt#chat-moderations

Source: https://docs.mistral.ai/api/#tag/chat_moderations_v1_chat_moderations_post

post /v1/chat/moderations

---

## 'completion_id': 0})]}))

**URL:** llms-txt#'completion_id':-0})]}))

py
refs = []
preds = []

for name in python_prompts:

# define user message
    user_message = prompt_template.format(
        task=python_prompts[name]["prompt"], name=name
    )

# run LLM
    response = run_mistral(user_message)

refs.append(python_prompts[name]["test"])
    preds.append([response])

**Examples:**

Example 1 (unknown):
```unknown
- Step 3: Calculate accuracy rate across test cases 

Now, we can go through all test cases, create a user message based on the prompt template, use the LLM to produce Python code, and evaluate the generated code for each test case.
```

---

## Create a agent that can be used within a conversation.

**URL:** llms-txt#create-a-agent-that-can-be-used-within-a-conversation.

Source: https://docs.mistral.ai/api/#tag/agents_api_v1_agents_create

---

## Create a conversation and append entries to it.

**URL:** llms-txt#create-a-conversation-and-append-entries-to-it.

Source: https://docs.mistral.ai/api/#tag/agents_api_v1_conversations_start_stream

post /v1/conversations#stream

---

## Define the messages for the chat

**URL:** llms-txt#define-the-messages-for-the-chat

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this image?"
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}" 
            }
        ]
    }
]

---

## Fim Completion

**URL:** llms-txt#fim-completion

Source: https://docs.mistral.ai/api/#tag/fim_completion_v1_fim_completions_post

post /v1/fim/completions

---

## Get the chat response

**URL:** llms-txt#get-the-chat-response

chat_response = client.chat.complete(
    model=model,
    messages=messages
)

---

## init the client but point it to TGI

**URL:** llms-txt#init-the-client-but-point-it-to-tgi

**Contents:**
  - Using a generate endpoint

client = OpenAI(api_key="-", base_url="http://127.0.0.1:8080/v1")
chat_response = client.chat.completions.create(
    model="-",
    messages=[
      {"role": "user", "content": "What is deep learning?"}
    ]
)

curl http://127.0.0.1:8080/v1/chat/completions \
    -X POST \
    -d '{
  "model": "tgi",
  "messages": [
    {
      "role": "user",
      "content": "What is deep learning?"
    }
  ]
}' \
    -H 'Content-Type: application/json'
python

**Examples:**

Example 1 (unknown):
```unknown
</TabItem>
  <TabItem value="curl" label="Using cURL" default>
```

Example 2 (unknown):
```unknown
</TabItem>
</Tabs>


### Using a generate endpoint

If you want more control over what you send to the server, you can use the `generate` endpoint. In this case, you're responsible of formatting the prompt with the correct template and stop tokens.

<Tabs>
  <TabItem value="python" label="Using Python" default>
```

---

## Letters Orders and Instructions December 1855\n\n**Hoag's Company, if any opportunity offers.**\n\nYou are to be particularly exact and careful in these pagineries, that there is no disgrace meet between the Returns and you Pay Roll, or those who will be strict examining into it hereafter.\n\nI am & c.\n\n*[Signed]*\nEff.

**URL:** llms-txt#letters-orders-and-instructions-december-1855\n\n**hoag's-company,-if-any-opportunity-offers.**\n\nyou-are-to-be-particularly-exact-and-careful-in-these-pagineries,-that-there-is-no-disgrace-meet-between-the-returns-and-you-pay-roll,-or-those-who-will-be-strict-examining-into-it-hereafter.\n\ni-am-&-c.\n\n*[signed]*\neff.

**Contents:**
- FAQ
- Introduction
- Getting started
  - Requesting access to the model
  - Querying the model
- Going further
- Introduction
- Getting started
  - Deploying the model
  - Querying the model

bash
curl https://api.mistral.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -d '{
    "model": "pixtral-12b-2409",
    "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text",
                     "text" : "Extract the text elements described by the user from the picture, and return the result formatted as a json in the following format : {name_of_element : [value]}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "From this restaurant bill, extract the bill number, item names and associated prices, and total price and return it as a string in a Json object"
                    },
                    {
                        "type": "image_url",
                        "image_url": "https://i.imghippo.com/files/kgXi81726851246.jpg"
                    }
                ]
            }
        ],
    "response_format": 
      {
        "type": "json_object"
      }
  }'

json
{'bill_number': '566548',
 'items': [{'item_name': 'BURGER - MED RARE', 'price': 10},
  {'item_name': 'WH/SUB POUTINE', 'price': 2},
  {'item_name': 'BURGER - MED RARE', 'price': 10},
  {'item_name': 'WH/SUB BSL - MUSH', 'price': 4},
  {'item_name': 'BURGER - MED WELL', 'price': 10},
  {'item_name': 'WH BREAD/NO ONION', 'price': 2},
  {'item_name': 'SUB POUTINE - MUSH', 'price': 2},
  {'item_name': 'CHK PESTO/BR', 'price': 9},
  {'item_name': 'SUB POUTINE', 'price': 2},
  {'item_name': 'SPEC OMELET/BR', 'price': 9},
  {'item_name': 'SUB POUTINE', 'price': 2},
  {'item_name': 'BSL', 'price': 8}],
 'total_price': 68}
python
        import boto3
        import os

region = os.environ.get("AWS_REGION")
        model_id = os.environ.get("AWS_BEDROCK_MODEL_ID")

bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=region)

user_msg = "Who is the best French painter? Answer in one short sentence."
        messages = [{"role": "user", "content": [{"text": user_msg}]}]
        temperature = 0.0
        max_tokens = 1024

params = {"modelId": model_id,
                  "messages": messages,
                  "inferenceConfig": {"temperature": temperature,
                                      "maxTokens": max_tokens}}

resp = bedrock_client.converse(**params)

print(resp["output"]["message"]["content"][0]["text"])
        shell
             aws bedrock-runtime converse \
             --region $AWS_REGION \
             --model-id $AWS_BEDROCK_MODEL_ID \
             --messages '[{"role": "user", "content": [{"text": "Who is the best French painter? Answer in one short sentence."}]}]'
            bash
        curl --location $AZUREAI_ENDPOINT/v1/chat/completions \
            --header  "Content-Type: application/json" \
            --header "Authorization: Bearer $AZURE_API_KEY" \
            --data '{
          "model": "azureai",
          "messages": [
            {
              "role": "user",
              "content": "Who is the best French painter? Answer in one short sentence."
            }
          ]
        }'
        python
        from mistralai_azure import MistralAzure
        import os

endpoint = os.environ.get("AZUREAI_ENDPOINT", "")
        api_key = os.environ.get("AZUREAI_API_KEY", "")

client = MistralAzure(azure_endpoint=endpoint,
                         azure_api_key=api_key)

resp = client.chat.complete(messages=[
            {
                "role": "user",
                "content": "Who is the best French painter? Answer in one short sentence."
            },
        ], model="azureai")

if resp:
            print(resp)
        typescript
        import { MistralAzure } from "@mistralai/mistralai-azure";

const client = new MistralAzure({
            endpoint: process.env.AZUREAI_ENDPOINT || "",
            apiKey: process.env.AZUREAI_API_KEY || ""
        });

async function chat_completion(user_msg: string) {
            const resp = await client.chat.complete({
                model: "azureai",
                messages: [
                    {
                        content: user_msg,
                        role: "user",
                    },
                ],
            });
            if (resp.choices && resp.choices.length > 0) {
                console.log(resp.choices[0]);
            }
        }

chat_completion("Who is the best French painter? Answer in one short sentence.");
        python
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from mistral_common.protocol.instruct.messages import UserMessage

import os
        import httpx

IBM_CLOUD_REGIONS = {
                "dallas": "us-south",
                "london": "eu-gb",
                "frankfurt": "eu-de",
                "tokyo": "jp-tok"
                }

IBM_CLOUD_PROJECT_ID = "xxx-xxx-xxx" # Replace with your project id

def get_iam_token(api_key: str) -> str:
            """
            Return an IAM access token generated from an API key.
            """

headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = f"apikey={api_key}&grant_type=urn:ibm:params:oauth:grant-type:apikey"
            resp = httpx.post(
                url="https://iam.cloud.ibm.com/identity/token",
                headers=headers,
                data=data,
            )
            token = resp.json().get("access_token")
            return token

def format_user_message(raw_user_msg: str) -> str:
            """
            Return a formatted prompt using the official Mistral tokenizer.
            """

tokenizer = MistralTokenizer.v3()  # Use v3 for Mistral Large
            tokenized = tokenizer.encode_chat_completion(
                ChatCompletionRequest(
                    messages=[UserMessage(content=raw_user_msg)], model="mistral-large"
                )
            )
            return tokenized.text

region = "frankfurt" # Define the region of your choice here
        api_key = os.environ["IBM_API_KEY"]
        access_token = get_iam_token(api_key=api_key)
        credentials = Credentials(url=f"https://{IBM_CLOUD_REGIONS[region]}.ml.cloud.ibm.com",
                                  token=access_token)

params = {GenParams.MAX_NEW_TOKENS: 256, GenParams.TEMPERATURE: 0.0}
        model_inference = ModelInference(
            project_id=IBM_CLOUD_PROJECT_ID,
            model_id="mistralai/mistral-large",
            params=params,
            credentials=credentials,
        )
        user_msg_content = "Who is the best French painter? Answer in one short sentence."
        resp = model_inference.generate_text(prompt=format_user_message(user_msg_content))
        print(resp)

bash
        echo $OUTSCALE_SERVER_URL/v1/chat/completions
        echo $OUTSCALE_MODEL_NAME
        curl --location $OUTSCALE_SRV_URL/v1/chat/completions \
          --header "Content-Type: application/json" \
          --header "Accept: application/json" \
          --data '{
              "model": "'"$OUTSCALE_MODEL_NAME"'",
              "temperature": 0,
              "messages": [
                {"role": "user", "content": "Who is the best French painter? Answer in one short sentence."}
              ],
              "stream": false
            }'
        python
        import os
        from mistralai import Mistral

client = Mistral(server_url=os.environ["OUTSCALE_SERVER_URL"])

resp = client.chat.complete(
            model=os.environ["OUTSCALE_MODEL_NAME"],
            messages=[
                {
                    "role": "user",
                    "content": "Who is the best French painter? Answer in one short sentence.",
                }
            ],
            temperature=0
        )

print(resp.choices[0].message.content)
        typescript
        import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({
            serverURL: process.env.OUTSCALE_SERVER_URL || ""
        });

const modelName = process.env.OUTSCALE_MODEL_NAME|| "";

async function chatCompletion(user_msg: string) {
            const resp = await client.chat.complete({
                model: modelName,
                messages: [
                    {
                        content: user_msg,
                        role: "user",
                    },
                ],
            });
            if (resp.choices && resp.choices.length > 0) {
                console.log(resp.choices[0]);
            }
        }

chatCompletion("Who is the best French painter? Answer in one short sentence.");
        bash
        curl --location $OUTSCALE_SERVER_URL/v1/fim/completions \
          --header "Content-Type: application/json" \
          --header "Accept: application/json" \
          --data '{
              "model": "'"$OUTSCALE_MODEL_NAME"'",
              "prompt": "def count_words_in_file(file_path: str) -> int:",
              "suffix": "return n_words",
              "stream": false
            }'
        python
        import os
        from mistralai import Mistral

client = Mistral(server_url=os.environ["OUTSCALE_SERVER_URL"])

resp = client.fim.complete(
            model = os.environ["OUTSCALE_MODEL_NAME"],
            prompt="def count_words_in_file(file_path: str) -> int:",
            suffix="return n_words"
        )

print(resp.choices[0].message.content)
       typescript
        import { Mistral} from "@mistralai/mistralai";

const client = new Mistral({
            serverURL: process.env.OUTSCALE_SERVER_URL || ""
        });

const modelName = "codestral-2405";

async function fimCompletion(prompt: string, suffix: string) {
            const resp = await client.fim.complete({
                model: modelName,
                prompt: prompt,
                suffix: suffix
            });
            if (resp.choices && resp.choices.length > 0) {
                console.log(resp.choices[0]);
            }
        }

fimCompletion("def count_words_in_file(file_path: str) -> int:",
                      "return n_words");
       SQL
    SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large2', 'Who is the best French painter? Answer in one short sentence.');
    python
        from snowflake.snowpark import Session
        from snowflake.ml.utils import connection_params
        from snowflake.cortex import Complete

# Start session (local execution only)
        params = connection_params.SnowflakeLoginOptions(connection_name="mistral")
        session = Session.builder.configs(params).create()

# Query the model
        prompt = "Who is the best French painter? Answer in one short sentence."
        completion = Complete(model="mistral-large2", prompt=prompt)
        print(completion)
        bash
        base_url="https://$GOOGLE_CLOUD_REGION-aiplatform.googleapis.com/v1/projects/$GOOGLE_CLOUD_PROJECT_ID/locations/$GOOGLE_CLOUD_REGION/publishers/mistralai/models"
        model_version="$VERTEX_MODEL_NAME@$VERTEX_MODEL_VERSION"
        url="$base_url/$model_version:rawPredict"

curl --location $url\
          --header "Content-Type: application/json" \
          --header "Authorization: Bearer $(gcloud auth print-access-token)" \
          --data '{
              "model": "'"$VERTEX_MODEL_NAME"'",
              "temperature": 0,
              "messages": [
                {"role": "user", "content": "Who is the best French painter? Answer in one short sentence."}
              ],
              "stream": false
            }'
        python
        import os
        from mistralai_gcp import MistralGoogleCloud

region = os.environ.get("GOOGLE_CLOUD_REGION")
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_NAME")
        model_name = os.environ.get("VERTEX_MODEL_NAME")
        model_version = os.environ.get("VERTEX_MODEL_VERSION")

client = MistralGoogleCloud(region=region, project_id=project_id)

resp = client.chat.complete(
            model = f"{model_name}-{model_version}",
            messages=[
                {
                    "role": "user",
                    "content": "Who is the best French painter? Answer in one short sentence.",
                }
            ],
        )

print(resp.choices[0].message.content)
        typescript
    import { MistralGoogleCloud } from "@mistralai/mistralai-gcp";

const client = new MistralGoogleCloud({
        region: process.env.GOOGLE_CLOUD_REGION || "",
        projectId: process.env.GOOGLE_CLOUD_PROJECT_ID || "",
    });

const modelName = process.env.VERTEX_MODEL_NAME|| "";
    const modelVersion = process.env.VERTEX_MODEL_VERSION || "";

async function chatCompletion(user_msg: string) {
        const resp = await client.chat.complete({
            model: modelName + "-" + modelVersion,
            messages: [
                {
                    content: user_msg,
                    role: "user",
                },
            ],
        });
        if (resp.choices && resp.choices.length > 0) {
            console.log(resp.choices[0]);
        }
    }

chatCompletion("Who is the best French painter? Answer in one short sentence.");
    bash
        VERTEX_MODEL_NAME=codestral
        VERTEX_MODEL_VERSION=2405

base_url="https://$GOOGLE_CLOUD_REGION-aiplatform.googleapis.com/v1/projects/$GOOGLE_CLOUD_PROJECT_ID/locations/$GOOGLE_CLOUD_REGION/publishers/mistralai/models"
        model_version="$VERTEX_MODEL_NAME@$VERTEX_MODEL_VERSION"
        url="$base_url/$model_version:rawPredict"

curl --location $url\
          --header "Content-Type: application/json" \
          --header "Authorization: Bearer $(gcloud auth print-access-token)" \
          --data '{
              "model":"'"$VERTEX_MODEL_NAME"'",
              "prompt": "def count_words_in_file(file_path: str) -> int:",
              "suffix": "return n_words",
              "stream": false
            }'
        python
        import os
        from mistralai_gcp import MistralGoogleCloud

region = os.environ.get("GOOGLE_CLOUD_REGION")
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_NAME")
        model_name = "codestral"
        model_version = "2405"

client = MistralGoogleCloud(region=region, project_id=project_id)

resp = client.fim.complete(
            model = f"{model_name}-{model_version}",
            prompt="def count_words_in_file(file_path: str) -> int:",
            suffix="return n_words"
        )

print(resp.choices[0].message.content)
        typescript
        import { MistralGoogleCloud } from "@mistralai/mistralai-gcp";

const client = new MistralGoogleCloud({
            region: process.env.GOOGLE_CLOUD_REGION || "",
            projectId: process.env.GOOGLE_CLOUD_PROJECT_ID || "",
        });

const modelName = "codestral";
        const modelVersion = "2405";

async function fimCompletion(prompt: string, suffix: string) {
            const resp = await client.fim.complete({
                model: modelName + "-" + modelVersion,
                prompt: prompt,
                suffix: suffix
            });
            if (resp.choices && resp.choices.length > 0) {
                console.log(resp.choices[0]);
            }
        }

fimCompletion("def count_words_in_file(file_path: str) -> int:",
                      "return n_words");
        bash
pip install cerebrium
cerebrium login
bash
cerebrium init mistral-vllm
toml
[cerebrium.deployment]
name = "mistral-vllm"
python_version = "3.11"
docker_base_image_url = "debian:bookworm-slim"
include = "[./*, main.py, cerebrium.toml]"
exclude = "[.*]"

[cerebrium.hardware]
cpu = 2
memory = 14.0
compute = "AMPERE_A10"
gpu_count = 1
provider = "aws"
region = "us-east-1"

[cerebrium.dependencies.pip]
sentencepiece = "latest"
torch = ">=2.0.0"
vllm = "latest"
transformers = ">=4.35.0"
accelerate = "latest"
xformers = "latest"
python
from vllm import LLM, SamplingParams
from huggingface_hub import login
from cerebrium import get_secret

**Examples:**

Example 1 (unknown):
```unknown
</details>

<details>
<summary><b>OCR with structured output</b></summary>

![](https://i.imghippo.com/files/kgXi81726851246.jpg)
```

Example 2 (unknown):
```unknown
Model output:
```

Example 3 (unknown):
```unknown
</details>

## FAQ

- **What is the price per image?**

    The price is calculated using the same pricing as input tokens per image, with each image being tokenized.

- **How many tokens correspond to an image and/or what is the maximum resolution?**

    Depending on the model and resolution, an image will be tokenized differently. Below is a summary.

    | Model | Max Resolution | ≈ Formula | ≈ N Max Tokens |
    | - | - | - | - |
    | Mistral Small 3.2 | 1540x1540 | `≈ (ResolutionX * ResolutionY) / 784` | ≈ 3025 |
    | Mistral Medium 3 | 1540x1540 | `≈ (ResolutionX * ResolutionY) / 784` | ≈ 3025 |
    | Mistral Small 3.1 | 1540x1540 | `≈ (ResolutionX * ResolutionY) / 784` | ≈ 3025 |
    | Pixtral Large | 1024x1024 | `≈ (ResolutionX * ResolutionY) / 256` | ≈ 4096 |
    | Pixtral 12B | 1024x1024 | `≈ (ResolutionX * ResolutionY) / 256` | ≈ 4096 |

    If the resolution of the image sent is higher than the maximum resolution of the model, the image will be downscaled to its maximum resolution. An error will be sent if the resolution is higher than **10000x10000**.

- **Can I fine-tune the image capabilities?**

    Yes, you can fine-tune pixtral-12b.

- **Can I use them to generate images?**

    No, they are designed to understand and analyze images, not to generate them.

- **What types of image files are supported?**
    
    We currently support the following image formats:

    - PNG (.png)
    - JPEG (.jpeg and .jpg)
    - WEBP (.webp) 
    - Non-animated GIF with only one frame (.gif) 

- **Is there a limit to the size of the image?**

    The current file size limit is 10Mb. 

- **What's the maximum number images per request?** 

    The maximum number images per request via API is 8.

- **What is the rate limit?**

    For information on rate limits, please visit https://console.mistral.ai/limits/.


[AWS Bedrock]
Source: https://docs.mistral.ai/docs/deployment/cloud/aws

## Introduction

Mistral AI's open and commercial models can be deployed on the AWS Bedrock cloud platform as
fully managed endpoints. AWS Bedrock is a serverless service so you don't have
to manage any infrastructure.

As of today, the following models are available:

- Mistral Large (24.07, 24.02)
- Mistral Small (24.02)
- Mixtral 8x7B
- Mistral 7B

For more details, visit the [models](../../../getting-started/models/models_overview/) page.

## Getting started

The following sections outline the steps to deploy and query a Mistral model on the
AWS Bedrock platform.

The following items are required:

- Access to an **AWS account** within a region that supports the AWS Bedrock service and 
  offers access to your model of choice: see 
  [the AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html) 
  for model availability per region.
- An AWS **IAM principal** (user, role) with sufficient permissions, see
  [the AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html)
  for more details.
- A local **code environment** set up with the relevant AWS SDK components, namely:
    - the AWS CLI: see [the AWS documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
      for the installation procedure.
    - the `boto3` Python library: see the [AWS documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html) 
      for the installation procedure.

### Requesting access to the model

Follow the instructions on
[the AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
to unlock access to the Mistral model of your choice.

### Querying the model

AWS Bedrock models are accessible through the Converse API.

Before running the examples below, make sure to sure to :

- Properly configure the authentication
credentials for your development environment. 
[The AWS documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)
provides an in-depth explanation on the required steps. 
- Create a Python virtual environment with the `boto3` package (version >= `1.34.131`).
- Set the following environment variables:
    - `AWS_REGION`: The region where the model is deployed (e.g. `us-west-2`),
    - `AWS_BEDROCK_MODEL_ID`: The model ID (e.g. `mistral.mistral-large-2407-v1:0`).

<Tabs>
    <TabItem value="python" label="Python">
```

Example 4 (unknown):
```unknown
</TabItem>
        <TabItem value="cli" label="AWS CLI">
```

---

## List all created conversations.

**URL:** llms-txt#list-all-created-conversations.

Source: https://docs.mistral.ai/api/#tag/agents_api_v1_conversations_list

get /v1/conversations

---

## make sure to install `llama-index` and `llama-index-llms-mistralai` in your Python enviornment

**URL:** llms-txt#make-sure-to-install-`llama-index`-and-`llama-index-llms-mistralai`-in-your-python-enviornment

**Contents:**
  - Devstral Integrations

from llama_index.core.llms import ChatMessage
from llama_index.llms.mistralai import MistralAI

api_key =  os.environ["MISTRAL_API_KEY"]
mistral_model = "codestral-latest"
messages = [
    ChatMessage(role="user", content="Write a function for fibonacci"),
]
MistralAI(api_key=api_key, model=mistral_model).chat(messages)
bash
pip install jupyterlab langchain-mistralai jupyter-ai pandas matplotlib
bash
jupyter lab
bash
[model.completion.http]
kind = "mistral/completion"
api_endpoint = "https://api.mistral.ai"
api_key = "secret-api-key"
bash

mkdir -p ~/.openhands && echo '{"language":"en","agent":"CodeActAgent","max_iterations":null,"security_analyzer":null,"confirmation_mode":false,"llm_model":"mistral/devstral-small-2507","llm_api_key":"'$MISTRAL_API_KEY'","remote_runtime_resource_factor":null,"github_token":null,"enable_default_condenser":true}' > ~/.openhands-state/settings.json

docker pull docker.all-hands.dev/all-hands-ai/runtime:0.48-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.48-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.48
```

For more information visit the [OpenHands github repo](https://github.com/All-Hands-AI/OpenHands) and their [documentation](https://docs.all-hands.dev/usage/llms/local-llms).

<details>
<summary><b>Integration with Cline</b></summary>

Cline is an autonomous coding agent operating right in your IDE, capable of creating/editing files, executing commands, using the browser, and more with your permission every step of the way.

<video width="100%" controls>
  <source src="/video/clinevideo.mov" type="video/mp4"/>
</video>

For more information visit the [Cline github repo](https://github.com/cline/cline).

[Annotations]
Source: https://docs.mistral.ai/docs/capabilities/document_ai/annotations

**Examples:**

Example 1 (unknown):
```unknown
Check out more details on using Instruct and Fill In Middle(FIM) with LlamaIndex in this [notebook](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/codestral.ipynb).

</details>

<details>
<summary><b>Integration with Jupyter AI</b></summary>

Jupyter AI seamlessly integrates Codestral into JupyterLab, offering users a streamlined and enhanced AI-assisted coding experience within the Jupyter ecosystem. This integration boosts productivity and optimizes users' overall interaction with Jupyter. 

To get started using Codestral and Jupyter AI in JupyterLab, first install needed packages in your Python environment:
```

Example 2 (unknown):
```unknown
Then launch Jupyter Lab:
```

Example 3 (unknown):
```unknown
Afterwards, you can select Codestral as your model of choice, input your Mistral API key, and start coding with Codestral!

<iframe width="560" height="315" width="100%" src="https://www.youtube.com/embed/jNUSTZwlq9M?si=plx_V19ZakgrniHy" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

</details>

<details>
<summary><b>Integration with JupyterLite</b></summary>

JupyterLite is a project that aims to bring the JupyterLab environment to the web browser, allowing users to run Jupyter directly in their browser without the need for a local installation.

You can try Codestral with JupyterLite in your browser:
[![lite-badge](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://jupyterlite.github.io/ai/lab/index.html)

<iframe width="560" height="315" width="100%" src="https://www.youtube.com/embed/edKyZSWy-Fw?si=pBzFV40vckyuCl6w" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

</details>

<details>
<summary><b>Integration with Tabby</b></summary>

Tabby is an open-source AI coding assistant. You can use Codestral for both code completion and chat via Tabby. 

To use Codestral in Tabby, configure your model configuration in `~/.tabby/config.toml` as follows.
```

Example 4 (unknown):
```unknown
You can check out [Tabby's documentation](https://tabby.tabbyml.com/docs/administration/model/#mistral--codestral) to learn more.  

<iframe width="560" height="315" width="100%" src="https://www.youtube.com/embed/ufHbMyC0oGA?si=kKlH8L3EtECMdtV7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

</details>

<details>
<summary><b>Integration with E2B</b></summary>

E2B provides open-source secure sandboxes for AI-generated code execution. 
With E2B, it is easy for developers to add code interpreting capabilities to AI apps using Codestral.

In the following examples, the AI agent performs a data analysis task on an uploaded CSV file, executes the AI-generated code by Codestral in the sandboxed environment by E2B, and returns a chart, saving it as a PNG file.

Python implementation ([cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/E2B_Code_Interpreting/codestral-code-interpreter-python)): 
<iframe width="560" height="315" width="100%" src="https://www.youtube.com/embed/26Wd-kC35Og?si=FgamyNZdzW--6iR7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

JS implementation ([cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/E2B_Code_Interpreting/codestral-code-interpreter-js)):
<iframe width="560" height="315" width="100%" src="https://www.youtube.com/embed/3M1_79U9RZE?si=YlTWN2chAxUhxHfr" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

</details>

### Devstral Integrations

<details>
<summary><b>Integration with Open Hands</b></summary>

OpenHands is an open-source scaffolding tool designed for building AI agents focused on software development. It offers a comprehensive framework for creating and managing these agents that can modify code, run commands, browse the web, call APIs, and even copy code snippets from StackOverflow.

<iframe width="560" height="315" width="100%" src="https://www.youtube.com/embed/oV9tAkS2Xic?si=gERKTfB-hFsSzk7f" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

After creating a Mistral AI account, you can use the following commands to start the OpenHands Docker container:
```

---

## Reprocess a document.

**URL:** llms-txt#reprocess-a-document.

**Contents:**
  - Objects
- Agent Creation
  - Creating an Agent
  - Updating an Agent
- Conversations
  - Starting a Conversation
  - Continue a Conversation
  - Retrieve Conversations
  - Restart Conversation
  - Streaming Output

Source: https://docs.mistral.ai/api/#tag/libraries_documents_reprocess_v1

post /v1/libraries/{library_id}/documents/{document_id}/reprocess

[Agents & Conversations]
Source: https://docs.mistral.ai/docs/agents/agents_and_conversations

We introduce three new main objects that our API makes use of:
- **Agents**: A set of pre-selected values to augment model abilities, such as tools, instructions, and completion parameters.
- **Conversation**: A history of interactions, past events and entries with an assistant, such as messages and tool executions, Conversations can be started by an Agent or a Model.
- **Entry**: An action that can be created by the user or an assistant. It brings a more flexible and expressive representation of interactions between a user and one or multiple assistants. This allows for more control over describing events.

*You can also leverage all the features of Agents and Conversations without the need to create an Agent. This means you can query our API without creating an Agent, from using the built-in Conversations features to the built-in Connectors.*

To find all details visit our [Agents](https://docs.mistral.ai/api/#tag/beta.agents) and [Conversations](https://docs.mistral.ai/api/#tag/beta.conversations) API spec.

When creating an Agent, there are multiple parameters and values that need to be set in advance. These are:
- `model`: The model your agent will use among our available models for chat completion.
- `description`: The agent description, related to the task it must accomplish or the use case at stake.
- `name`: The name of your agent.
- `instructions` *optional*: The main instructions of the agent, also known as the system prompt. This must accurately describe the main task of your agent.
- `tools` *optional*: A list of tools the model can make use of. There are currently different `types` of tools:
  - `function`: User-defined tools, with similar usage to the standard function calling used with chat completion.
  - `web_search`/`web_search_premium`: Our built-in tool for web search.
  - `code_interpreter`: Our built-in tool for code execution.
  - `image_generation`: Our built-in tool for image generation.
  - `document_library`: Our built-in RAG tool for knowledge grounding and search on custom data.
- `completion_args` *optional*: Standard chat completion sampler arguments. All chat completion arguments are accepted.

### Creating an Agent
<Tabs groupId="code">
  <TabItem value="python" label="python" default>

When creating an agent, you will receive an Agent object with an agent ID. You can then use that ID to have conversations.

Here is an example of a Web Search Agent using our built-in tool:

You can find more information [here](../connectors/websearch).
  </TabItem>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

<details>
    <summary><b>JSON Output</b></summary>

### Updating an Agent

After creation, you can update the Agent with new settings if needed. The arguments are the same as those used when creating an Agent.  
The result is a new `version` of the Agent with the new settings, you can this way have the previous and new versions available.

#### Create a new Version
Create a new `version` of the Agent, will be used by default.
<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

<details>
    <summary><b>JSON Output</b></summary>

#### Change Version
Change manually the version of the Agent.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

<details>
    <summary><b>JSON Output</b></summary>

Once your agent is created, you can **start** conversations at any point while keeping the same conversation persistent. You first start a conversation by providing:
- `agent_id`: The ID of the agent, created during the Agent creation.
- `inputs`: The message to start the conversation with. It can be either a string with the first user message or question, or the history of messages.

Creating a Conversation will return a conversation ID.

To **continue** the conversation and append the exchanges as you go, you provide two values:
- `conversation_id`: The ID created during the conversation start or append that maps to the internally stored conversation history.
- `inputs`: The next message or reply. It can be either a string or a list of messages.

A new Conversation ID is provided at each append.

You can also **opt out** from the automatic storing with `store=False`; this will make the new history not being stored on our cloud.

We also provide the parameter `handoff_execution`, which currently has two modes: `server` or `client`.
- `server`: Runs the handoff as expected internally on our cloud servers; this is the default setting.
- `client`: When a handoff is triggered, a response is provided directly to the user, enabling them to handle the handoff with control.

For more information regarding handoffs visit [this section](../agents/handoffs).

### Starting a Conversation
<Tabs groupId="code">
  <TabItem value="python" label="python" default>

Both options are equivalent.

Without an Agent, querying Conversations could look like so:

<TabItem value="typescript" label="typescript">

Both options are equivalent.

Without an Agent, querying Conversations could look like so:

<TabItem value="curl" label="curl">

<details>
    <summary><b>JSON Output</b></summary>

### Continue a Conversation
You can continue the conversation; the history is stored when using the correct conversation ID.
<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

<details>
    <summary><b>JSON Output</b></summary>

### Retrieve Conversations
You can retrieve conversations; both all available already created and the details of each.

Retrieve conversations available:
<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

<details>
    <summary><b>JSON Output</b></summary>

Retrieve details from a specific conversation:
<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

<details>
    <summary><b>JSON Output</b></summary>

Retrieve entries and history from a specific conversation:
<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

<details>
    <summary><b>JSON Output</b></summary>

Retrieve all messages from a specific conversation:
<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

<details>
    <summary><b>JSON Output</b></summary>

### Restart Conversation

You can continue a conversation from any given entry from the history of entries:
<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

<details>
    <summary><b>JSON Output</b></summary>

**Note**: You can only restart conversations on which you used the `append()` method at least once.

### Streaming Output
You can also stream the outputs, both when starting a conversation, continuing or restarting a previous one.
<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

For each streaming operation, you should use the following snippet of code:

<TabItem value="curl" label="curl">

When streaming, you will have specific indexes for specific content types during a stream. These include:
- `conversation.response.started`: The start of a conversation response.
- `conversation.response.done`: The response is done and finished.
- `conversation.response.error`: An error occurred.
- `message.output.delta`: Chunk of content, usually tokens corresponding to the model reply.
- `tool.execution.started`: A tool execution has started.
- `tool.execution.done`: A tool has finished executing.
- `agent.handoff.started`: The handoff to a different agent has started.
- `agent.handoff.done`: The handoff was concluded.
- `function.call.delta`: Chunk of content, usually tokens corresponding to the function tool call.

<details>
    <summary><b>Example</b></summary>

[Agents Function Calling]
Source: https://docs.mistral.ai/docs/agents/agents_function_calling

The core of an agent relies on its tool usage capabilities, enabling it to use and call tools and workflows depending on the task it must accomplish.

Built into our API, we provide [connector](../connectors/connectors) tools such as `websearch`, `code_interpreter`, `image_generation` and `document_library`. However, you can also use standard function tool calling by defining a JSON schema for your function.

You can also leverage our MCP Orchestration to implement local Function Calling, visit our [Local MCP docs](../mcp/#step-4-register-mcp-client) for further details.

For more information regarding function calling, we recommend to visit our [function calling docs](../../capabilities/function_calling).

### Creating an Agent with Function Calling

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

We need to define our function that we want our model to call when needed, in this case, the function is a dummy for demonstration purposes.

Once defined, we provide a Shema corresponding to the same function.

<TabItem value="typescript" label="typescript">

We need to define our function that we want our model to call when needed, in this case, the function is a dummy for demonstration purposes.

Once defined, we provide a Shema corresponding to the same function.

<TabItem value="curl" label="curl">

<details>
    <summary><b>Output</b></summary>

### Using an Agent with Function Calling

<Tabs groupId="code">
  <TabItem value="python" label="python" default>
Then, to use it, we start a conversation or continue a previously existing one.

<details>
    <summary><b>Output</b></summary>

The model will output either an answer, or a function call, we need to detect and return the result of the expected function.

<TabItem value="typescript" label="typescript">
Then, to use it, we start a conversation or continue a previously existing one.

<details>
    <summary><b>Output</b></summary>

The model will output either an answer, or a function call, we need to detect and return the result of the expected function.

First, let's add the following imports:

Then, we check whether or not a function call was triggered:

<TabItem value="curl" label="curl">

For starting a conversation:

For continuing a conversation:

<details>
    <summary><b>Output</b></summary>

[Agents Introduction]
Source: https://docs.mistral.ai/docs/agents/agents_introduction

## What are AI agents?

<div style={{ textAlign: 'center' }}>
  <img
    src="/img/agent_overview.png"
    alt="agent_introduction"
    width="600"
    style={{ borderRadius: '15px' }}
  />
</div>

AI agents are autonomous systems powered by large language models (LLMs) that, given high-level instructions, can plan, use tools, carry out processing steps, and take actions to achieve specific goals. These agents leverage advanced natural language processing capabilities to understand and execute complex tasks efficiently and can even collaborate with each other to achieve more sophisticated outcomes.

Our Agents and Conversations API allows developers to build such agents, leveraging multiple features such as:
- Multiple mutlimodal models available, **text and vision models**.
- **Persistent state** across conversations.
- Ability to have conversations with **base models**, **a single agent**, and **multiple agents**.
- Built-in connector tools for **code execution**, **web search**, **image generation** and **document library** out of the box.
- **Handoff capability** to use different agents as part of a workflow, allowing agents to call other agents.
- Features supported via our chat completions endpoint are also supported, such as:
  - **Structured Outputs**
  - **Document Understanding**
  - **Tool Usage**
  - **Citations**

## More Information
- [Agents & Conversations](../agents_basics): Basic explanations and code snippets around our Agents and Conversations API.
- [Connectors](../connectors/connectors): Make your tools accessible directly to any Agents.
  - [Websearch](../connectors/websearch): In-depth explanation of our web search built-in connector tool.
  - [Code Interpreter](../connectors/code_interpreter): In-depth explanation of our code interpreter for code execution built-in connector tool.
  - [Image Generation](../connectors/image_generation): In-depth explanation of our image generation built-in connector tool.
  - [Document Library (Beta)](../connectors/document_library): A RAG built-in connector enabling Agents to access a library of documents.
- [MCP](../mcp): How to use [MCP](../../capabilities/function_calling) (Model Context Protocol) servers with Agents.
- [Function Calling](../function_calling): How to use [Function calling](../../capabilities/function_calling) with Agents.
- [Handoffs](../handoffs): Relay tasks and use other agents as tools in agentic workflows.

## Cookbooks
For more information and guides on how to use our Agents, we have the following cookbooks:
- [Github Agent](https://github.com/mistralai/cookbook/tree/main/mistral/agents/agents_api/github_agent)
- [Linear Tickets](https://github.com/mistralai/cookbook/tree/main/mistral/agents/agents_api/prd_linear_ticket)
- [Financial Analyst](https://github.com/mistralai/cookbook/tree/main/mistral/agents/agents_api/financial_analyst)
- [Travel Assistant](https://github.com/mistralai/cookbook/tree/main/mistral/agents/agents_api/travel_assistant)
- [Food Diet Companion](https://github.com/mistralai/cookbook/tree/main/mistral/agents/agents_api/food_diet_companion)

- **Which models are supported?**

Currently, only `mistral-medium-latest` and `mistral-large-latest` are supported, but we will soon enable it for more models.

[Code Interpreter]
Source: https://docs.mistral.ai/docs/agents/connectors/code_interpreter

Code Interpreter adds the capability to safely execute code in an isolated container, this built-in [connector](../connectors) tool allows Agents to run code at any point on demand, practical to draw graphs, data analysis, mathematical operations, code validation, and much more.

<div style={{ textAlign: 'center' }}>
  <img
    src="/img/code_interpreter_connector.png"
    alt="code_interpreter_graph"
    width="400"
    style={{ borderRadius: '15px' }}
  />
</div>

## Create a Code Interpreter Agent
You can create an agent with access to our code interpreter by providing it as one of the tools.  
Note that you can still add more tools to the agent, the model is free to run code or not on demand.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

<details>
    <summary><b>Output</b></summary>

As for other agents, when creating one you will receive an agent id corresponding to the created agent that you can use to start a conversation.

### Conversations with Code Interpreter
Now that we have our coding agent ready, we can at any point make use of it to run code.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

For explanation purposes, lets take a look at the output in a readable JSON format.

### Explanation of the Outputs 
There are 3 main entries in the `outputs` of our request:

- **`message.output`**: This entry corresponds to the initial response from the assistant, indicating that it can help generate the first 20 Fibonacci numbers.

- **`tool.execution`**: This entry corresponds to the execution of the code interpreter tool. It includes metadata about the execution, such as:
  - `name`: The name of the tool, which in this case is `code_interpreter`.
  - `object`: The type of object, which is `entry`.
  - `type`: The type of entry, which is `tool.execution`.
  - `created_at` and `completed_at`: Timestamps indicating when the tool execution started and finished.
  - `id`: A unique identifier for the tool execution.
  - `info`: This section contains additional information specific to the tool execution. For the `code_interpreter` tool, the `info` section includes:
    - `code`: The actual code that was executed. In this example, it contains a Python function `fibonacci(n)` that generates the first `n` numbers in the Fibonacci sequence and a call to this function to get the first 20 Fibonacci numbers.
    - `code_output`: The output of the executed code, which is the list of the first 20 Fibonacci numbers.

- **`message.output`**: This entry corresponds to the final response from the assistant, providing the first 20 values of the Fibonacci sequence.

[Connectors Overview]
Source: https://docs.mistral.ai/docs/agents/connectors/connectors_overview

Connectors are tools that Agents can call at any given point. They are deployed and ready for the agents to leverage to answer questions on demand.

<div style={{ textAlign: 'center' }}>
  <img
    src="/img/connectors_graph.png"
    alt="connectors_graph"
    width="800"
    style={{ borderRadius: '15px' }}
  />
</div>

They are also available for users to use them directly via Conversations without the Agent creation step!

## General Usage
<Tabs groupId="code">
  <TabItem value="python" label="python" default>
You can either create an Agent with the desired tools:

Or call our conversations API directly:

<TabItem value="typescript" label="typescript">
    
You can either create an Agent with the desired tools:

Or call our conversations API directly:

<TabItem value="curl" label="curl">
You can either create an Agent with the desired tools:

Or call our conversations API directly:

Currently, our API has 4 built-in Connector tools, here you can find how to use them in details:
- [Websearch](../websearch)
- [Code Interpreter](../code_interpreter)
- [Image Generation](../image_generation)
- [Document Library](../document_library)

[Document Library]
Source: https://docs.mistral.ai/docs/agents/connectors/document_library

Document Library is a built-in [connector](../connectors) tool that enables agents to access documents from Mistral Cloud.

<div style={{ textAlign: 'center' }}>
  <img
    src="/img/document_library_connector.png"
    alt="document_library_graph"
    width="400"
    style={{ borderRadius: '15px' }}
  />
</div>

It is a built-in RAG capability that enhances your agents' knowledge with the data you have uploaded.

You can manage your libraries using the Mistral AI API, we recommend taking a look at the [API spec](https://docs.mistral.ai/api/#tag/beta.libraries.documents) for the details. Below are some examples of how to interact with libraries and documents.

### Listing Libraries

You can list your existing libraries and their documents.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

<details>
    <summary><b>Output</b></summary>

### Listing Documents in a Library

To list documents in a specific library:

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

You can list and search documents in a library if required.

### Creating a New Library

You can create and manage new document libraries directly via our API.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

<details>
    <summary><b>Contents</b></summary>

When generated, the library will contain different kinds of information. This information is updated and generated when files are added. Specifically, `generated_name` and `generated_description` will be constantly updated and kept up to date.

### Listing Documents in a New Library

Each library, has a set of documents that belongs to it.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="typescript" label="typescript">

<TabItem value="curl" label="curl">

At first, a new library will not have any documents inside.

### Uploading a Document

You can upload and remove documents from a library.

#### Upload
<Tabs groupId="code">
  <TabItem value="python" label="python" default>

```python
from mistralai.models import File

**Examples:**

Example 1 (py):
```py
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key)

simple_agent = client.beta.agents.create(
    model="mistral-medium-2505",
    description="A simple Agent with persistent state.",
    name="Simple Agent"
)
```

Example 2 (py):
```py
websearch_agent = client.beta.agents.create(
    model="mistral-medium-2505",
    description="Agent able to search information over the web, such as news, weather, sport results...",
    name="Websearch Agent",
    instructions="You have the ability to perform web searches with `web_search` to find up-to-date information.",
    tools=[{"type": "web_search"}],
    completion_args={
        "temperature": 0.3,
        "top_p": 0.95,
    }
)
```

Example 3 (typescript):
```typescript
dotenv.config();

const apiKey = process.env.MISTRAL_API_KEY;

const client = new Mistral({ apiKey: apiKey });

async function main() {
  let websearchAgent = await client.beta.agents.create({
    model: "mistral-medium-latest",
    name: "WebSearch Agent",
    instructions: "Use your websearch abilities when answering requests you don't know.",
    description: "Agent able to fetch new information on the web.",
    tools: [{ type: "web_search" }],
  });
}
```

Example 4 (bash):
```bash
curl --location "https://api.mistral.ai/v1/agents" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
     "model": "mistral-medium-latest",
     "name": "Simple Agent",
     "description": "A simple Agent with persistent state."
  }'
```

---

## Restart a conversation starting from a given entry.

**URL:** llms-txt#restart-a-conversation-starting-from-a-given-entry.

Source: https://docs.mistral.ai/api/#tag/agents_api_v1_conversations_restart_stream

post /v1/conversations/{conversation_id}/restart#stream

---

## Retrieve all entries in a conversation.

**URL:** llms-txt#retrieve-all-entries-in-a-conversation.

Source: https://docs.mistral.ai/api/#tag/agents_api_v1_conversations_history

get /v1/conversations/{conversation_id}/history

---

## Retrieve all messages in a conversation.

**URL:** llms-txt#retrieve-all-messages-in-a-conversation.

Source: https://docs.mistral.ai/api/#tag/agents_api_v1_conversations_messages

get /v1/conversations/{conversation_id}/messages

---

## Retrieve a conversation information.

**URL:** llms-txt#retrieve-a-conversation-information.

Source: https://docs.mistral.ai/api/#tag/agents_api_v1_conversations_get

get /v1/conversations/{conversation_id}

---

## Tokenize a list of messages

**URL:** llms-txt#tokenize-a-list-of-messages

**Contents:**
- v3 tokenizer
  - Our tokenization vocabulary
  - Run our tokenizer in Python
- Use cases
  - NLP tasks
  - Tokens count
- Mistral AI Crawlers
  - MistralAI-User

tokenized = tokenizer.encode_chat_completion(
    ChatCompletionRequest(
        tools=[
            Tool(
                function=Function(
                    name="get_current_weather",
                    description="Get the current weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use. Infer this from the users location.",
                            },
                        },
                        "required": ["location", "format"],
                    },
                )
            )
        ],
        messages=[
            UserMessage(content="What's the weather like today in Paris"),
        ],
        model=model_name,
    )
)
tokens, text = tokenized.tokens, tokenized.text

<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "format": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The temperature unit to use. Infer this from the users location."}}, "required": ["location", "format"]}}}][/AVAILABLE_TOOLS][INST]What's the weather like today in Paris[/INST]

<unk>
<s>
</s>
[INST]
[/INST]
[TOOL_CALLS]
[AVAILABLE_TOOLS]
[/AVAILABLE_TOOLS]
[TOOL_RESULTS]
[/TOOL_RESULTS]

▁▁
▁▁▁▁
▁t
in
er
...
벨
ゼ
梦
python
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    UserMessage,
    ToolMessage
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.tool_calls import Function, Tool, ToolCall, FunctionCall
from mistral_common.protocol.instruct.request import ChatCompletionRequest

tokenizer_v3 = MistralTokenizer.v3()
python
tokenized = tokenizer_v3.encode_chat_completion(
    ChatCompletionRequest(
        tools=[
            Tool(
                function=Function(
                    name="get_current_weather",
                    description="Get the current weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use. Infer this from the users location.",
                            },
                        },
                        "required": ["location", "format"],
                    },
                )
            )
        ],
        messages=[
            UserMessage(content="What's the weather like today in Paris"),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="VvvODy9mT",
                        function=FunctionCall(
                            name="get_current_weather",
                            arguments='{"location": "Paris, France", "format": "celsius"}',
                        ),
                    )
                ],
            ),
            ToolMessage(
                tool_call_id="VvvODy9mT", name="get_current_weather", content="22"
            ),
            AssistantMessage(
                content="The current temperature in Paris, France is 22 degrees Celsius.",
            ),
            UserMessage(content="What's the weather like today in San Francisco"),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="fAnpW3TEV",
                        function=FunctionCall(
                            name="get_current_weather",
                            arguments='{"location": "San Francisco", "format": "celsius"}',
                        ),
                    )
                ],
            ),
            ToolMessage(
                tool_call_id="fAnpW3TEV", name="get_current_weather", content="20"
            ),
        ],
        model="test",
    )
)

tokens, text = tokenized.tokens, tokenized.text

'<s>[INST] What\'s the weather like today in Paris[/INST][TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Paris, France", "format": "celsius"}, "id": "VvvODy9mT"}]</s>[TOOL_RESULTS] {"call_id": "VvvODy9mT", "content": 22}[/TOOL_RESULTS] The current temperature in Paris, France is 22 degrees Celsius.</s>[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "format": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The temperature unit to use. Infer this from the users location."}}, "required": ["location", "format"]}}}][/AVAILABLE_TOOLS][INST] What\'s the weather like today in San Francisco[/INST][TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "San Francisco", "format": "celsius"}, "id": "fAnpW3TEV"}]</s>[TOOL_RESULTS] {"call_id": "fAnpW3TEV", "content": 20}[/TOOL_RESULTS]'
```
To count the number of tokens, run `len(tokens)` and we get 302 tokens.

## Use cases
### NLP tasks

As we mentioned earlier, tokenization is a crucial first step in natural language processing (NLP) tasks. Once we have tokenized our text, we can use those tokens to create text embeddings, which are dense vector representations of the text. These embeddings can then be used for a variety of NLP tasks, such as text classification, sentiment analysis, and machine translation.

Mistral's embedding API simplifies this process by combining the tokenization and embedding steps into one. With this API, we can easily create text embeddings for a given text, without having to separately tokenize the text and create embeddings from the tokens.

If you're interested in learning more about how to use Mistral's embedding API, be sure to check out our [embedding guide](/capabilities/embeddings/overview), which provides detailed instructions and examples.

Mistral AI's LLM API endpoints charge based on the number of tokens in the input text.

To help you estimate your costs, our tokenization API makes it easy to count the number of tokens in your text. Simply run `len(tokens)` as shown in the example above to get the total number of tokens in the text, which you can then use to estimate your cost based on our pricing information.

[Mistral AI Crawlers]
Source: https://docs.mistral.ai/docs/robots

## Mistral AI Crawlers

Mistral AI employs web crawlers ("robots") and user agents to execute tasks for its products, either automatically or upon user request. To facilitate webmasters in managing how their sites and content interact with AI, Mistral AI utilizes specific robots.txt tags.

MistralAI-User is for user actions in LeChat. When users ask LeChat a question, it may visit a web page to help answer and include a link to the source in its response. MistralAI-User governs which sites these user requests can be made to. It is not used for crawling the web in any automatic fashion, nor to crawl content for generative AI training.

Full user-agent string: Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; MistralAI-User/1.0; +https://docs.mistral.ai/robots)

Published IP addresses: https://mistral.ai/mistralai-user-ips.json

**Examples:**

Example 1 (unknown):
```unknown
Here is the output of “text”, which is a debug representation for you to inspect.
```

Example 2 (unknown):
```unknown
To count the number of tokens, run `len(tokens)` and we get 128 tokens.

## v3 tokenizer 

Our v3 tokenizer uses the Byte-Pair Encoding (BPE) with SentencePiece, which is an open-source tokenization library to build our tokenization vocabulary.

In BPE, the tokenization process starts by treating each byte in a text as a separate token. 
Then, it iteratively adds new tokens to the vocabulary for the most frequent pair of tokens currently appearing in the corpus. For example, if the most frequent pair of tokens is "th" + "e", then a new token "the" will be created and occurrences of "th"+"e" will be replaced with the new token "the". This process continues until no more replacements can be made.

### Our tokenization vocabulary
Our tokenization vocabulary is released in the https://github.com/mistralai/mistral-common/tree/main/tests/data folder. Let’s take a look at the vocabulary of our v3 tokenizer. 

#### Vocabulary size
Our vocabulary consists of 32k vocab + 768 control tokens. The 32k vocab includes 256 bytes and 31,744 characters and merged characters. 

#### Control tokens 
Our vocabulary starts with 10 control tokens, which are special tokens we use in the encoding process to represent specific instructions or indicators:
```

Example 3 (unknown):
```unknown
#### Bytes
After the control token slots, we have 256 bytes in the vocabulary. A byte is a unit of digital information that consists of 8 bits. Each bit can represent one of two values, either 0 or 1. A byte can therefore represent 256 different values.
```

Example 4 (unknown):
```unknown
Any character, regardless of the language or symbol, can be represented by a sequence of one or more bytes. When a word is not present in the vocabulary, it can still be represented by the bytes that correspond to its individual characters. This is important for handling unknown words and characters. 

#### Characters and merged characters
And finally, we have the characters and merged characters in the vocabulary. The order of the tokens are determined by the frequency of these tokens in the data that was used to train the model, with the most frequent ones in the beginning of the vocabulary. For example, two spaces “▁”, four spaces “▁▁▁▁”, “_t”, “in”, and “er” were found to be the most common tokens we trained on. As we move further down the vocabulary list, the tokens become less frequent. Towards the end of the vocabulary file, you might find less common characters such as Chinese and Korean characters. These characters are less frequent because they were encountered less often in the training data, not because they are less used in general.
```

---
