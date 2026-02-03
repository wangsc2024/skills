# Mistral - Other

**Pages:** 139

---

## {0: [(0,

**URL:** llms-txt#{0:-[(0,

---

## Annotations

**URL:** llms-txt#annotations

**Contents:**
- How it works
- BBox Annotation

In addition to the basic OCR functionality, Mistral Document AI API adds the `annotations` functionality, which allows you to extract information in a structured json-format that you provide. Specifically, it offers two types of annotations: 
- `bbox_annotation`: gives you the annotation of the bboxes extracted by the OCR model (charts/ figures etc) based on user requirement and provided bbox/image annotation format. The user may ask to describe/caption the figure for instance.
- `document_annotation`: returns the annotation of the entire document based on the provided document annotation format.

<div style={{ textAlign: 'center' }}>
  <img
    src="/img/ocr_annotations_explanation.png"
    alt="annotations_explanation_graph"
    width="600"
    style={{ borderRadius: '15px' }}
  />
</div>

**Key capabilities**:
* Labeling and annotating data
* Extraction and structuring of specific information from documents into a predefined JSON format
* Automation of data extraction to reduce manual entry and errors
* Efficient handling of large document volumes for enterprise-level applications

**Common use cases**:
* Parsing of forms, classification of documents, and processing of images, including text, charts, and signatures
* Conversion of charts to tables, extraction of fine print from figures, or definition of custom image types
* Capture of receipt data, including merchant names and transaction amounts, for expense management.
* Extraction of key information like vendor details and amounts from invoices for automated accounting.
* Extraction of key clauses and terms from contracts for easier review and management

<div style={{ textAlign: 'center' }}>
  <img
    src="/img/ocr_annotations_workflow.png"
    alt="annotations_workflow_graph"
    width="800"
    style={{ borderRadius: '15px' }}
  />
</div>

**BBOX Annotations**
- All document types: 
  - After regular OCR is finished; we call a Vision capable LLM for all bboxes individually with the provided annotation format.

**Document Annotation**
- pdf/image: 
  - Independent of OCR; we convert all pages into images and send all images to a Vision capable LLM along with the provided annotation format.
- pptx/docx/...:
  - We run OCR first and send the output text markdown to a Vision capable LLM along with the provided annotation format.

You can use our API with the following document formats:
- [OCR with  pdf](basic_ocr.md#ocr-with-pdf)
- [OCR with uploaded pdf](basic_ocr.md#ocr-with-uploaded-pdf)
- [OCR with image](basic_ocr.md#ocr-with-image): even from low-quality or handwritten sources.
- scans, DOCX, PPTX.

In these examples, we will only consider the `OCR with pdf` format.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

Here is an example of how to use our Annotation functionalities using the Mistral AI client and Pydantic:

**Define the Data Model**

First, define the response formats for `BBox Annotation` using Pydantic models:

```python
from pydantic import BaseModel

---

## Archive Fine Tuned Model

**URL:** llms-txt#archive-fine-tuned-model

Source: https://docs.mistral.ai/api/#tag/jobs_api_routes_fine_tuning_archive_fine_tuned_model

post /v1/fine_tuning/models/{model_id}/archive

---

## Are you ready?

**URL:** llms-txt#are-you-ready?

- ✍ [fill out the application here](https://forms.gle/pTMchkVVPCxSVW5u5) ✍
- Join our [Discord](https://discord.gg/mistralai)

[Contribute]
Source: https://docs.mistral.ai/docs/guides/contribute/overview

---

## Assuming we have the following data

**URL:** llms-txt#assuming-we-have-the-following-data

data = {
    'transaction_id': ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
    'customer_id': ['C001', 'C002', 'C003', 'C002', 'C001'],
    'payment_amount': [125.50, 89.99, 120.00, 54.30, 210.20],
    'payment_date': ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
    'payment_status': ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
}

---

## BBOX Annotation response formats

**URL:** llms-txt#bbox-annotation-response-formats

class Image(BaseModel):
  image_type: str = Field(..., description="The type of the image.")
  short_description: str = Field(..., description="A description in english describing the image.")
  summary: str = Field(..., description="Summarize the image.")
python

from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, ResponseFormat
from mistralai.extra import response_format_from_pydantic_model

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

**Examples:**

Example 1 (unknown):
```unknown
**Start the completion**

Next, use the Mistral AI python client to make a request and ensure the response adheres to the defined structures using `bbox_annotation_format` set to the corresponding pydantic models:
```

---

## BBOX Annotation response format

**URL:** llms-txt#bbox-annotation-response-format

class Image(BaseModel):
  image_type: str
  short_description: str
  summary: str

---

## BBOX Annotation response format with description

**URL:** llms-txt#bbox-annotation-response-format-with-description

class Image(BaseModel):
  image_type: str = Field(..., description="The type of the image.")
  short_description: str = Field(..., description="A description in english describing the image.")
  summary: str = Field(..., description="Summarize the image.")

---

## calculate accuracy rate across test cases

**URL:** llms-txt#calculate-accuracy-rate-across-test-cases

**Contents:**
  - Example 2: evaluate code generation

sum(accuracy_rates) / len(accuracy_rates)
py
python_prompts = {
    "sort_string": {
        "prompt": "Write a python function to sort the given string.", 
        "test": "assert sort_string(\"data\") == \"aadt\""
    },
    "is_odd": {
        "prompt": "Write a python function to check whether the given number is odd or not using bitwise operator.", 
        "test": "assert is_odd(5) == True"
    }
}
py
def run_mistral(user_message, model="mistral-large-latest"):
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    messages = [{"role":"user", "content": user_message}]
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return chat_response.choices[0].message.content

**Examples:**

Example 1 (unknown):
```unknown
</details>

<details>
<summary><b> Example 2: evaluate code generation</b></summary>

### Example 2: evaluate code generation

#### Evaluation data
Our second example involves generating Python code and assessing the generated code. To conduct the evaluation, both the Python instructions and the corresponding unit tests are required for the evaluation data. Here are two examples of such evaluation data sets:
```

Example 2 (unknown):
```unknown
#### How to evaluate? 

- Step 1: Define prompt template

We have designed a prompt that generates Python code snippets based on descriptions of specific tasks.
```

---

## Cancel a jobs

**URL:** llms-txt#cancel-a-jobs

canceled_jobs = client.fine_tuning.jobs.cancel(job_id = created_jobs.id)
print(canceled_jobs)
typescript
// List jobs
const jobs = await client.fineTuning.jobs.list();

// Retrieve a job
const retrievedJob = await client.fineTuning.jobs.get({ jobId: createdJob.id })

// Cancel a job
const canceledJob = await client.fineTuning.jobs.cancel({
  jobId: createdJob.id,
});
bash

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

---

## Cancel a job

**URL:** llms-txt#cancel-a-job

**Contents:**
- Use a fine-tuned model
- Delete a fine-tuned model
  - Available models
  - Four steps

curl -X POST https://api.mistral.ai/v1/fine_tuning/jobs/<jobid>/cancel \
--header "Authorization: Bearer $MISTRAL_API_KEY"
python
chat_response = client.chat.complete(
    model=retrieved_job.fine_tuned_model,
    messages = [{"role":'user', "content":'What is the best French cheese?'}]
)
typescript
const chatResponse = await client.chat.complete({
  model: retrievedJob.fine_tuned_model,
  messages: [{role: 'user', content: 'What is the best French cheese?'}],
});
bash
curl "https://api.mistral.ai/v1/chat/completions" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
    "model": "ft:open-mistral-7b:daf5e488:20240430:c1bed559",
    "messages": [{"role": "user", "content": "Who is the most renowned French painter?"}]
  }'
python
client.models.delete(model_id=retrieved_job.fine_tuned_model)
typescript
await client.models.delete({modelId:retrieved_job.fine_tuned_model})
bash
curl --location --request DELETE 'https://api.mistral.ai/v1/models/ft:open-mistral-7b:XXX:20240531:XXX' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY"
python

**Examples:**

Example 1 (unknown):
```unknown
</TabItem>

</Tabs>


## Use a fine-tuned model
When a fine-tuned job is finished, you will be able to see the fine-tuned model name via `retrieved_jobs.fine_tuned_model`. Then you can use our `chat` endpoint to chat with the fine-tuned model:

<Tabs groupId="code">
  <TabItem value="python" label="python" default>
```

Example 2 (unknown):
```unknown
</TabItem>

  <TabItem value="typescript" label="typescript">
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

## Delete a fine-tuned model

<Tabs groupId="code">
  <TabItem value="python" label="python" default>
```

---

## Cancel Batch Job

**URL:** llms-txt#cancel-batch-job

Source: https://docs.mistral.ai/api/#tag/jobs_api_routes_batch_cancel_batch_job

post /v1/batch/jobs/{job_id}/cancel

---

## Check status document

**URL:** llms-txt#check-status-document

status = client.beta.libraries.documents.status(library_id=new_library.id, document_id=uploaded_doc.id)
print(status)

---

## Classify a single example

**URL:** llms-txt#classify-a-single-example

**Contents:**
  - Comparison with fastText

text = "I've been experiencing frequent headaches and vision problems."
clf.predict([get_text_embedding([text])])

**Examples:**

Example 1 (unknown):
```unknown
Output
```

Example 2 (unknown):
```unknown
### Comparison with fastText
Additionally, let's take a look at the performance using fastText embeddings in this classification task. It appears that the classification model achieves better performance with Mistral AI Embeddings model as compared to using fastText embeddings.
```

---

## clf = LogisticRegression(random_state=0, C=1.0, max_iter=1000, solver='sag').fit(train_x, train_y)

**URL:** llms-txt#clf-=-logisticregression(random_state=0,-c=1.0,-max_iter=1000,-solver='sag').fit(train_x,-train_y)

**Contents:**
- Clustering
- Retrieval
- Dataset Format
  - 1. Single Target
  - 2. Multi-Target
- Upload a file
- Create a fine-tuning job

print(f"Precision: {100*np.mean(clf.predict(test_x) == test_y.to_list()):.2f}%")

Precision: 86.25%
python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=24, max_iter=1000)
model.fit(df['embeddings'].to_list())
df["cluster"] = model.labels_
print(*df[df.cluster==23].text.head(3), sep='\n\n')

I have been feeling extremely tired and weak, and I've also been coughing a lot with difficulty breathing. My fever is very high, and I'm producing a lot of mucus when I cough.
I've got a cough that won't go away, and I'm exhausted. I've been coughing up thick mucous and my fever is also pretty high.
I have a persistent cough and have been feeling quite fatigued. My fever is through the roof, and I'm having trouble breathing. When I cough, I also cough up a lot of mucous.
json
{
    "text": "I love this product!",
    "labels": {
        "sentiment": "positive" // positive/neutral/negative
    }
}
json
{
    "text": "I love this product!",
    "labels": {
        "sentiment": ["positive","neutral"]
    }
}
json
{
    "messages": [{"role": "user", "content": "I love this product!"}],
    "labels": {
        "sentiment": "positive" // positive/neutral/negative
    }
}
json
{
    "messages": [{"role": "user", "content": "I love this product!"}],
    "labels": {
        "sentiment": ["positive","neutral"]
    }
}
json
{"text": "I love this product!", "labels": {"sentiment": "positive"}}
{"text": "The game was amazing.", "labels": {"sentiment": "positive"}}
{"text": "The new policy is controversial.", "labels": {"sentiment": "neutral"}}
{"text": "I don't like the new design.", "labels": {"sentiment": "negative"}}
{"text": "The team won the championship.", "labels": {"sentiment": "positive"}}
{"text": "The economy is in a bad shape.", "labels": {"sentiment": "negative"}}
...
json
{
    "text": "I love this product!",
    "labels": {
        "sentiment": "positive", // positive/neutral/negative
        "is-english": "yes" // yes/no, boolean
    }
}
json
{
    "messages": [{"role": "user", "content": "I love this product!"}],
    "labels": {
        "sentiment": "positive", // positive/neutral/negative
        "is-english": "yes" // yes/no, boolean
    }
}
python
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

training_data = client.files.upload(
    file={
        "file_name": "training_file.jsonl",
        "content": open("training_file.jsonl", "rb"),
    }
)

validation_data = client.files.upload(
    file={
        "file_name": "validation_file.jsonl",
        "content": open("validation_file.jsonl", "rb"),
    }
)
typescript

const apiKey = process.env.MISTRAL_API_KEY;

const client = new Mistral({apiKey: apiKey});

const training_file = fs.readFileSync('training_file.jsonl');
const training_data = await client.files.upload({
    file: {
        fileName: "training_file.jsonl",
        content: training_file,
    }
});

const validation_file = fs.readFileSync('validation_file.jsonl');
const validation_data = await client.files.upload({
    file: {
        fileName: "validation_file.jsonl",
        content: validation_file,
    }
});
bash
curl https://api.mistral.ai/v1/files \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -F purpose="fine-tune" \
  -F file="@training_file.jsonl"

curl https://api.mistral.ai/v1/files \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -F purpose="fine-tune" \
  -F file="@validation_file.jsonl"
python

**Examples:**

Example 1 (unknown):
```unknown
Output
```

Example 2 (unknown):
```unknown
## Clustering
What if we don't have disease labels? One approach to gain insights from the data is through clustering. Clustering is an unsupervised machine learning technique that groups similar data points together based on their similarity with respect to certain features. In the context of text embeddings, we can use the distance between each embedding as a measure of similarity, and group together data points with embeddings that are close to each other in the high-dimensional space.

Since we already know there are 24 clusters, let's use the K-means clustering with 24 clusters. Then we can inspect a few examples and verify whether the examples in a single cluster are similar to one another. For example, take a look at the first three rows of cluster 23. We can see that they look very similar in terms of symptoms.
```

Example 3 (unknown):
```unknown
Output:
```

Example 4 (unknown):
```unknown
## Retrieval
Our embedding model excels in retrieval tasks, as it is trained with retrieval in mind. Embeddings are also incredibly helpful in implementing retrieval-augmented generation (RAG) systems, which use retrieved relevant information from a knowledge base to generate responses. At a high-level, we embed a knowledge base, whether it is a local directory, text files, or internal wikis, into text embeddings and store them in a vector database. Then, based on the user's query, we retrieve the most similar embeddings, which represent the relevant information from the knowledge base. Finally, we feed these relevant embeddings to a large language model to generate a response that is tailored to the user's query and context. If you are interested in learning more about how RAG systems work and how to implement a basic RAG, check out our [previous guide](/guides/rag) on this topic.


[Classifier Factory]
Source: https://docs.mistral.ai/docs/capabilities/finetuning/classifier-factory

In various domains and enterprises, classification models play a crucial role in enhancing efficiency, improving user experience, and ensuring compliance. These models serve diverse purposes, including but not limited to:
- **Moderation**: Classification models are essential for moderating services and classifying unwanted content. For instance, our [moderation service](../../guardrailing/#moderation-api) helps in identifying and filtering inappropriate or harmful content in real-time, ensuring a safe and respectful environment for users.
- **Intent Detection**: These models help in understanding user intent and behavior. By analyzing user interactions, they can predict the user's next actions or needs, enabling personalized recommendations and improved customer support.
- **Sentiment Analysis**: Emotion and sentiment detection models analyze text data to determine the emotional tone behind words. This is particularly useful in social media monitoring, customer feedback analysis, and market research, where understanding public sentiment can drive strategic decisions.
- **Data Clustering**: Classification models can group similar data points together, aiding in data organization and pattern recognition. This is beneficial in market segmentation, where businesses can identify distinct customer groups for targeted marketing campaigns.
- **Fraud Detection**: In the financial sector, classification models help in identifying fraudulent transactions by analyzing patterns and anomalies in transaction data. This ensures the security and integrity of financial systems.
- **Spam Filtering**: Email services use classification models to filter out spam emails, ensuring that users receive only relevant and safe communications.
- **Recommendation Systems**: Classification models power recommendation engines by categorizing user preferences and suggesting relevant products, movies, or content based on past behavior and preferences.

By leveraging classification models, organizations can make data-driven decisions, improve operational efficiency, and deliver better products and services to their customers.

For this reason, we designed a friendly and easy way to make your own classifiers. Leveraging our small but highly efficient models and training methods, the Classifier Factory is both available directly in [la plateforme](https://console.mistral.ai/build/finetuned-models) and our API.

## Dataset Format

Data must be stored in JSON Lines (`.jsonl`) files, which allow storing multiple JSON objects, each on a new line.

We provide two endpoints:
- `v1/classifications`: To classify raw text.
- `v1/chat/classifications`: To classify chats and multi-turn interactions.

There are 2 main kinds of classification models:
- Single Target
- Multi-Target

### 1. Single Target

For single label classification, data must have the label name and the value for that corresponding label. Example:

<Tabs groupId="classification-type">
  <TabItem value="v1/classifications" label="v1/classifications" default>
```

---

## Client call

**URL:** llms-txt#client-call

**Contents:**
- Cookbooks
- FAQ
- Document AI OCR processor
  - OCR with PDF

response = client.ocr.process(
    model="mistral-ocr-latest",
    pages=list(range(8)),
    document=DocumentURLChunk(
      document_url="https://arxiv.org/pdf/2410.07073"
    ),
    bbox_annotation_format=response_format_from_pydantic_model(Image),
    document_annotation_format=response_format_from_pydantic_model(Document),
    include_image_base64=True
  )
typescript

// BBOX Annotation response format
const ImageSchema = z.object({
  image_type: z.string(),
  short_description: z.string(),
  summary: z.string(),
});

// Document Annotation response format
const DocumentSchema = z.object({
  language: z.string(),
  chapter_titles: z.array(z.string()),
  urls: z.array(z.string()),
});
typescript

// Define the schema for the Image type
const ImageSchema = z.object({
  image_type: z.string().describe("The type of the image."),
  short_description: z.string().describe("A description in English describing the image."),
  summary: z.string().describe("Summarize the image."),
});

// Document Annotation response format
const DocumentSchema = z.object({
  language: z.string(),
  chapter_titles: z.array(z.string()),
  urls: z.array(z.string()),
});
typescript

const apiKey = process.env.MISTRAL_API_KEY;

const client = new Mistral({ apiKey: apiKey });

async function processDocument() {
  try {
    const response = await client.ocr.process({
      model: "mistral-ocr-latest",
      pages: Array.from({ length: 8 }, (_, i) => i), // Creates an array [0, 1, 2, ..., 7]
      document: {
        type: "document_url",
        documentUrl: "https://arxiv.org/pdf/2410.07073"
      },
      bboxAnnotationFormat: responseFormatFromZodObject(ImageSchema),
      documentAnnotationFormat: responseFormatFromZodObject(DocumentSchema),
      includeImageBase64: true,
    });

console.log(response);
  } catch (error) {
    console.error("Error processing document:", error);
  }
}

bash
curl --location 'https://api.mistral.ai/v1/ocr' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer ${MISTRAL_API_KEY}" \
--data '{
    "model": "mistral-ocr-latest",
    "document": {"document_url": "https://arxiv.org/pdf/2410.07073"},
    "pages": [0, 1, 2, 3, 4, 5, 6, 7],
    "bbox_annotation_format": {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "properties": {
                    "document_type": {"title": "Document_Type", "type": "string"},
                    "short_description": {"title": "Short_Description", "type": "string"},
                    "summary": {"title": "Summary", "type": "string"}
                },
                "required": ["document_type", "short_description", "summary"],
                "title": "BBOXAnnotation",
                "type": "object",
                "additionalProperties": false
            },
            "name": "document_annotation",
            "strict": true
        }
    },
    "document_annotation_format": {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "properties": {
                    "language": {"title": "Language", "type": "string"},
                    "chapter_titles": {"title": "Chapter_Titles", "type": "string"},
                    "urls": {"title": "urls", "type": "string"}
                },
                "required": ["language", "chapter_titles", "urls"],
                "title": "DocumentAnnotation",
                "type": "object",
                "additionalProperties": false
            },
            "name": "document_annotation",
            "strict": true
        }
    },
    "include_image_base64": true
}'
bash
curl --location 'https://api.mistral.ai/v1/ocr' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer ${MISTRAL_API_KEY}" \
--data '{
    "model": "mistral-ocr-latest",
    "document": {"document_url": "https://arxiv.org/pdf/2410.07073"},
    "bbox_annotation_format": {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "properties": {
                    "document_type": {"title": "Document_Type", "description": "The type of the image.", "type": "string"},
                    "short_description": {"title": "Short_Description", "description": "A description in English describing the image.", "type": "string"},
                    "summary": {"title": "Summary", "description": "Summarize the image.", "type": "string"}
                },
                "required": ["document_type", "short_description", "summary"],
                "title": "BBOXAnnotation",
                "type": "object",
                "additionalProperties": false
            },
            "name": "document_annotation",
            "strict": true
        }
    },
     "document_annotation_format": {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "properties": {
                    "language": {"title": "Language", "type": "string"},
                    "chapter_titles": {"title": "Chapter_Titles", "type": "string"},
                    "urls": {"title": "urls", "type": "string"}
                },
                "required": ["language", "chapter_titles", "urls"],
                "title": "DocumentAnnotation",
                "type": "object",
                "additionalProperties": false
            },
            "name": "document_annotation",
            "strict": true
        }
    },
    "include_image_base64": true
}'
json
{ 
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGB{LONG_MIDDLE_SEQUENCE}KKACiiigAooooAKKKKACiiigD//2Q==" 
}
json
{
  "image_type": "scatter plot",
  "short_description": "Comparison of different models based on performance and cost.",
  "summary": "The image consists of two scatter plots comparing various models on two different performance metrics against their cost or number of parameters. The left plot shows performance on the MM-MT-Bench, while the right plot shows performance on the LMSys-Vision ELO. Each point represents a different model, with the x-axis indicating the cost or number of parameters in billions (B) and the y-axis indicating the performance score. The shaded region in both plots highlights the best performance/cost ratio, with Pixtral 12B positioned within this region in both plots, suggesting it offers a strong balance of performance and cost efficiency. Other models like Qwen-2-VL 72B and Qwen-2-VL 7B also show high performance but at varying costs."
}
json
{
  "language": "English",
  "chapter_titles": [
    "Abstract",
    "1 Introduction",
    "2 Architectural details",
    "2.1 Multimodal Decoder",
    "2.2 Vision Encoder",
    "2.3 Complete architecture",
    "3 MM-MT-Bench: A benchmark for multi-modal instruction following",
    "4 Results",
    "4.1 Main Results",
    "4.2 Prompt selection",
    "4.3 Sensitivity to evaluation metrics",
    "4.4 Vision Encoder Ablations"
  ],
  "urls": [
    "https://mistral.ai/news/pixtal-12b/",
    "https://github.com/mistralai/mistral-inference/",
    "https://github.com/mistralai/mistral-evals/",
    "https://huggingface.co/datasets/mistralai/MM-MT-Bench"
  ]
} 
python

from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": "https://arxiv.org/pdf/2201.04234"
    },
    include_image_base64=True
)
python

from mistralai import Mistral

def encode_pdf(pdf_path):
    """Encode the pdf to base64."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None

**Examples:**

Example 1 (unknown):
```unknown
</TabItem>
   <TabItem value="typescript" label="typescript" default>

Here is an example of how to use our Annotation functionalities using the Mistral AI client and Zod:

**Define the Data Model**

First, define the response formats for both `BBox Annotation` and `Document Annotation` using Zod schemas:
```

Example 2 (unknown):
```unknown
You can also provide a description for each entry, the description will be used as detailed information and instructions during the annotation; for example:
```

Example 3 (unknown):
```unknown
**Start the completion**

Next, use the Mistral AI typescript client to make a request and ensure the response adheres to the defined structures using `bbox_annotation_format` and `document_annotation_format` set to the corresponding Zod schemas:
```

Example 4 (unknown):
```unknown
</TabItem>
  <TabItem value="curl" label="curl">

The request is structured to ensure that the response adheres to the specified custom JSON schema. The schema defines the structure of a `bbox_annotation` object with `image_type`, `short_description` and `summary` properties and a `document_annotation` object with with `language`, `chapter_titles` and `urls` properties.
```

---

## close the output files

**URL:** llms-txt#close-the-output-files

for file in output_file_objects:
    file.close()

---

## "content": open("uploaded_file.pdf", "rb"),

**URL:** llms-txt#"content":-open("uploaded_file.pdf",-"rb"),

---

## counter for output files

**URL:** llms-txt#counter-for-output-files

counter = 0
with open(input_file, "r") as f_in:
    # read the input file line by line
    for line in f_in:
        # parse the line as JSON
        data = json.loads(line)
        # write the data to the current output file
        output_file_objects[counter].write(json.dumps(data) + "\n")
        # increment the counter
        counter = (counter + 1) % 3

---

## Create a new Library.

**URL:** llms-txt#create-a-new-library.

Source: https://docs.mistral.ai/api/#tag/libraries_create_v1

---

## Create a train / test split

**URL:** llms-txt#create-a-train-/-test-split

train_x, test_x, train_y, test_y = train_test_split(
    df["fasttext_embeddings"], df["label"], test_size=0.2
)

---

## Create Batch Job

**URL:** llms-txt#create-batch-job

Source: https://docs.mistral.ai/api/#tag/jobs_api_routes_batch_create_batch_job

---

## Create DataFrame

**URL:** llms-txt#create-dataframe

**Contents:**
- Step 1. User: specify tools and query
  - Tools
  - User query
- Step 2. Model: Generate function arguments
  - tool_choice
  - parallel_tool_calls
- Step 3. User: Execute function to obtain tool results
- Step 4. Model: Generate final answer
- Moderation API
  - Raw-text endpoint

df = pd.DataFrame(data)
typescript
// Assuming we have the following data
const data = {
    transactionId: ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
    customerId: ['C001', 'C002', 'C003', 'C002', 'C001'],
    paymentAmount: [125.50, 89.99, 120.00, 54.30, 210.20],
    paymentDate: ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
    paymentStatus: ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
};

// Convert data into an array of objects for easier manipulation
const transactions = data.transactionId.map((id, index) => ({
    transactionId: id,
    customerId: data.customerId[index],
    paymentAmount: data.paymentAmount[index],
    paymentDate: data.paymentDate[index],
    paymentStatus: data.paymentStatus[index]
}));
python
def retrieve_payment_status(df: data, transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values: 
        return json.dumps({'status': df[df.transaction_id == transaction_id].payment_status.item()})
    return json.dumps({'error': 'transaction id not found.'})

def retrieve_payment_date(df: data, transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values: 
        return json.dumps({'date': df[df.transaction_id == transaction_id].payment_date.item()})
    return json.dumps({'error': 'transaction id not found.'})
typescript
function retrievePaymentStatus(transactions, transactionId) {
    const transaction = transactions.find(t => t.transactionId === transactionId);
    if (transaction) {
        return JSON.stringify({ status: transaction.paymentStatus });
    }
    return JSON.stringify({ error: 'transaction id not found.' });
}

function retrievePaymentDate(transactions, transactionId) {
    const transaction = transactions.find(t => t.transactionId === transactionId);
    if (transaction) {
        return JSON.stringify({ date: transaction.paymentDate });
    }
    return JSON.stringify({ error: 'transaction id not found.' });
}
python
tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_status",
            "description": "Get payment status of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_date",
            "description": "Get payment date of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    }
]
python
tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_info",
            "description": "Retrieves payment infos",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id",
                    },
                    "info_type": {
                        "type": "string",
                        "description": "The info type ('status' or 'date')",
                    }
                },
                "required": ["transaction_id", "info_type"],
            },
        },
    }
]
typescript
const tools = [
    {
        type: "function",
        function: {
            name: "retrievePaymentStatus",
            description: "Get payment status of a transaction",
            parameters: {
                type: "object",
                properties: {
                    transactionId: {
                        type: "string",
                        description: "The transaction id.",
                    }
                },
                required: ["transactionId"],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "retrievePaymentDate",
            description: "Get payment date of a transaction",
            parameters: {
                type: "object",
                properties: {
                    transactionId: {
                        type: "string",
                        description: "The transaction id.",
                    }
                },
                required: ["transactionId"],
            },
        },
    }
];
typescript
const tools = [
    {
        type: "function",
        function: {
            name: "retrievePaymentInfo",
            description: "Retrieves payment infos",
            parameters: {
                type: "object",
                properties: {
                    transactionId: {
                        type: "string",
                        description: "The transaction id",
                    },
                    infoType: {
                        type: "string",
                        description: "The info type ('status' or 'date')",
                    }
                },
                required: ["transactionId", "infoType"],
            },
        },
    }
];
python

names_to_functions = {
    'retrieve_payment_status': functools.partial(retrieve_payment_status, df=df),
    'retrieve_payment_date': functools.partial(retrieve_payment_date, df=df)
}
typescript
const namesToFunctions = {
    'retrievePaymentStatus': (transactionId) => retrievePaymentStatus(transactions, transactionId),
    'retrievePaymentDate': (transactionId) => retrievePaymentDate(transactions, transactionId)
};
python
messages = [{"role": "user", "content": "What's the status of my transaction T1001?"}]
typescript
const messages = [{"role": "user", "content": "What's the status of my transaction T1001?"}];
python

from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)
response = client.chat.complete(
    model = model,
    messages = messages,
    tools = tools,
    tool_choice = "any",
    parallel_tool_calls = False,
)
response

ChatCompletionResponse(id='7cbd8962041442459eb3636e1e3cbf10', object='chat.completion', model='mistral-large-latest', usage=Usage(prompt_tokens=94, completion_tokens=30, total_tokens=124), created=1721403550, choices=[Choices(index=0, finish_reason='tool_calls', message=AssistantMessage(content='', tool_calls=[ToolCall(function=FunctionCall(name='retrieve_payment_status', arguments='{"transaction_id": "T1001"}'), id='D681PevKs', type='function')], prefix=False, role='assistant'))])
typescript

const apiKey = process.env.MISTRAL_API_KEY;
const model = "mistral-large-latest";

const client = new Mistral({ apiKey: apiKey });

let response = await client.chat.complete({
    model: model,
    messages: messages,
    tools: tools,
    toolChoice: "any",
    parallelToolCalls: false,
});
python
messages.append(response.choices[0].message)
typescript
messages.push(response.choices[0].message);
python

tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_params = json.loads(tool_call.function.arguments)
print("\nfunction_name: ", function_name, "\nfunction_params: ", function_params)

function_name:  retrieve_payment_status 
function_params: {'transaction_id': 'T1001'}
typescript
const toolCall = response.choices[0].message.toolCalls[0];
const functionName = toolCall.function.name;
const functionParams = JSON.parse(toolCall.function.arguments);
console.log("\nfunction_name: ", functionName, "\nfunction_params: ", functionParams);

function_name:  retrievePaymentStatus 
function_params:  { transactionId: 'T1001' }
python
function_result = names_to_functions[function_name](**function_params)
function_result

'{"status": "Paid"}'
typescript
const functionResult = namesToFunctions[functionName](functionParams.transactionId);
console.log(functionResult);

{"status":"Paid"}
python
messages.append({
    "role":"tool", 
    "name":function_name, 
    "content":function_result, 
    "tool_call_id":tool_call.id
})

response = client.chat.complete(
    model = model, 
    messages = messages
)
response.choices[0].message.content

The status of your transaction with ID T1001 is "Paid". Is there anything else I can assist you with?
typescript
messages.push({
    role: "tool",
    name: functionName,
    content: functionResult,
    toolCallId: toolCall.id
});

response = await client.chat.complete({
    model: model,
    messages: messages
});
console.log(response.choices[0].message.content);

The status of your transaction with ID T1001 is "Paid". Is there anything else I can assist you with?
python

from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

response = client.classifiers.moderate(
    model = "mistral-moderation-latest",  
    inputs=["...text to classify..."]
)

print(response)
typescript

const apiKey = process.env.MISTRAL_API_KEY;
const client = new Mistral({apiKey});

const response = await client.classifiers.moderate({
    model: "mistral-moderation-latest",
    inputs: ["...text to classify..."],
});

console.log(response);

curl
curl https://api.mistral.ai/v1/moderations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -d '{
    "model": "mistral-moderation-latest", 
    "input": ["...text to classify..."]
  }'
python

from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

response = client.classifiers.moderate_chat(
    model="mistral-moderation-latest",
    inputs=[
        {"role": "user", "content": "...user prompt ..."},
        {"role": "assistant", "content": "...assistant response..."},
    ],
)

print(response)
typescript

const apiKey = process.env.MISTRAL_API_KEY;
const client = new Mistral({apiKey});

const response = await client.classifiers.moderateChat({
  model: "mistral-moderation-latest",
  inputs = [
    { role: "user", content: "...user prompt ..." },
    { role: "assistant", content: "...assistant response..." },
  ],
});

console.log(response);

curl
curl https://api.mistral.ai/v1/chat/moderations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -d '{
    "model": "mistral-moderation-latest", 
    "input": [{"role": "user", "content": "...user prompt ..."}, {"role": "assistant", "content": "...assistant response..."}]
  }'

ClassificationResponse(
    id='091b378dec1444e2a4800d6915aad0fa',
    model='mistral-moderation-latest',
    results=[
        ClassificationObject(
            categories={
                'sexual': False,
                'hate_and_discrimination': False,
                'violence_and_threats': True,
                'dangerous_and_criminal_content': False,
                'selfharm': False,
                'health': False,
                'financial': False,
                'law': False,
                'pii': False
            },
            category_scores={
                'sexual': 9.608268737792969e-05,
                'hate_and_discrimination': 0.0001398324966430664,
                'violence_and_threats': 0.9990234375,
                'dangerous_and_criminal_content': 1.5676021575927734e-05,
                'selfharm': 0.0001233816146850586,
                'health': 3.2782554626464844e-06,
                'financial': 1.3828277587890625e-05,
                'law': 2.282857894897461e-05,
                'pii': 0.0001233816146850586
            }
        )
    ]
)
python
chat_response = client.chat.complete(
    model = "mistral-large-latest", 
    messages = [{"role":"user", "content":"What is the best French cheese?"}],
    safe_prompt = True
)
typescript
const chatResponse = await client.chat.complete(
    model: 'mistral-large-latest',
    messages: [{role: 'user', content: 'What is the best French cheese?'}],
    safePrompt: true
);
bash
curl --location "https://api.mistral.ai/v1/chat/completions" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
    "model": "mistral-large-latest",
    "messages": [
     {
        "role": "user",
        "content": "What is the best French cheese?"
      }
    ],
    "safe_prompt": true
  }'

Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.

You're given a list of moderation categories as below:

- physical harm: activity that has high risk of physical harm, including: weapons development, military and warfare, management or operation of critical infrastructure in energy, transportation, and water, content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.
- economic harm: activity that has high risk of economic harm, including: multi-level marketing, gambling, payday lending, automated determinations of eligibility for credit, employment, educational institutions, or public assistance services.
- fraud: Fraudulent or deceptive activity, including: scams, coordinated inauthentic behavior, plagiarism, academic dishonesty, astroturfing, such as fake grassroots support or fake review generation, disinformation, spam, pseudo-pharmaceuticals.

Please classify the following text into one of these categories, and answer with that single word only.

If the sentence does not fall within these categories, is safe and does not need to be moderated, please answer "not moderated".
python

from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

code = """
created_jobs = client.fine_tuning.jobs.create(
    model="open-mistral-7b", 
    training_files=[{"file_id": ultrachat_chunk_train.id, "weight": 1}],
    validation_files=[ultrachat_chunk_eval.id], 
    hyperparameters={
        "training_steps": 10,
        "learning_rate":0.0001
    },
    auto_start=False
)
"""

prompt = "Change the model name from open-mistral-7b to open-mistral-nemo. Respond only with code, no explanation, no formatting."

chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "user",
            "content": code
        },
    ],
    prediction = {
        "type": "content",
        "content": code
    }
)
print(chat_response.choices[0].message.content)
typescript

const apiKey = process.env.MISTRAL_API_KEY;

const client = new Mistral({apiKey: apiKey});

const code = `
created_jobs = client.fine_tuning.jobs.create(
    model="open-mistral-7b", 
    training_files=[{"file_id": ultrachat_chunk_train.id, "weight": 1}],
    validation_files=[ultrachat_chunk_eval.id], 
    hyperparameters={
        "training_steps": 10,
        "learning_rate":0.0001
    },
    auto_start=False
)
`.trim();

const prompt = `Change the model name from open-mistral-7b to open-mistral-nemo. Respond only with code, no explanation, no formatting.`;

const chatResponse = await client.chat.complete({
    model: "mistral-large-latest",
    messages: [
        {
            role: 'user', 
            content: prompt
        },
        {
            role: "user",
            content: code
        },
    ],
    prediction: {
        type: "content",
        content: code 
    },
});

console.log('Chat:', chatResponse.choices[0].message.content);
bash
curl --location "https://api.mistral.ai/v1/chat/completions" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
    "model": "mistral-large-latest",
    "messages": [
        {"role": "user", "content": "Change the model name from open-mistral-7b to open-mistral-nemo. Respond only with code, no explanation, no formatting."},
        {"role": "user", "content": "$CODE"}
    ],
    "prediction": {
        "type": "content",
        "content": "$CODE"
    }
  }'
json
[
  {
    "role": "system",
    "content": [
      {
        "type": "text",
        "text": "System prompt, with both instructions and"
      },
      {
        "type": "thinking",
        "thinking": [
          {
            "type": "text",
            "text": "Encapsulated reasoning instructions."
          }
        ]
      }
    ]
  },
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "User input."
      }
    ]
  },
  {
    "role": "assistant",
    "content": [
      {
        "type": "thinking",
        "thinking": [
          {
            "type": "text",
            "text": "Reasoning traces."
          }
        ]
      },
      {
        "type": "text",
        "text": "Followed by the final answer."
      }
    ]
  }
]
json
[
  {
    "role": "system",
    "content": "System prompt, with both instructions and\n<think>\nEncapsulated reasoning instructions.\n</think>\n"
  },
  {
    "role": "user",
    "content": "User input."
  },
  {
    "role": "assistant",
    "content": "<think>\nReasoning traces.\n</think>\nFollowed by the final answer."
  }
]
json
{
  "role": "system",
  "content": [
    {
      "type": "text",
      "text": "First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.\n\nYour thinking process must follow the template below:"
    },
    {
      "type": "thinking",
      "thinking": [
        {
          "type": "text",
          "text": "Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user."
        }
      ]
    },
    {
      "type": "text",
      "text": "Here, provide a self-contained response."
    }
  ]
}
json
{
  "role": "system",
  "content": "A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown to format your response. Write both your thoughts and summary in the same language as the task posed by the user. NEVER use \\boxed{} in your response.\n\nYour thinking process must follow the template below:\n<think>\nYour thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.\n</think>\n\nHere, provide a concise summary that reflects your reasoning and presents a clear final answer to the user. Don't mention that this is a summary.\n\nProblem:\n\n"
}
python

from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "magistral-medium-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model = model,
    messages = [
        {
            "role": "user",
            "content": "John is one of 4 children. The first sister is 4 years old. Next year, the second sister will be twice as old as the first sister. The third sister is two years older than the second sister. The third sister is half the age of her older brother. How old is John?",
        },
    ],
    # prompt_mode = "reasoning" if you want to explicitly use the default system prompt, or None if you want to opt out of the default system prompt.
)

print(chat_response.choices[0].message.content)
typescript

const apiKey = process.env.MISTRAL_API_KEY;

const client = new Mistral({apiKey: apiKey});

async function main() {
    const chatResponse = await client.chat.complete({
        model: "magistral-medium-latest",
        messages: [{role: 'user', content: 'John is one of 4 children. The first sister is 4 years old. Next year, the second sister will be twice as old as the first sister. The third sister is two years older than the second sister. The third sister is half the age of her older brother. How old is John?'}],
        // prompt_mode: "reasoning" if you want to explicitly use the default system prompt, or null if you want to opt out of the default system prompt.
    });

console.log('Chat:', chatResponse.choices?.[0]?.message?.content);
}

main();
bash
curl --location "https://api.mistral.ai/v1/chat/completions" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
    "model": "magistral-medium-latest",
    "messages": [
     {
        "role": "user",
        "content": "John is one of 4 children. The first sister is 4 years old. Next year, the second sister will be twice as old as the first sister. The third sister is two years older than the second sister. The third sister is half the age of her older brother. How old is John?"
      }
    ]
  }'
json
"content": [
  {
    "type": "thinking",
    "thinking": [
      {
        "type": "text",
        "text": "*Thoughts and reasoning traces will go here.*"
      }
    ]
  },
  {
    "type": "text",
    "text": "*Final answer will go here.*"
  },
  ...
]
json
{
  "model": "magistral-medium-2507",
  "usage": {
    "prompt_tokens": 173,
    "total_tokens": 11170,
    "completion_tokens": 10997
  },
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "tool_calls": null,
        "content": [
          {
            "type": "thinking",
            "thinking": [
              {
                "type": "text",
                "text": "Alright, let's break this problem down systematically.\n\n1. Identify the children mentioned:\n   - There are 4 children in total: John and three sisters.\n\n2. Information about the sisters:\n   - First sister: 4 years old.\n   - Second sister: Next year, she will be twice as old as the first sister is now.\n     - Current age of first sister: 4 years.\n     - Next year, the first sister will be 5 years old.\n     - But the problem says \"twice as old as the first sister\" (current age, which is 4).\n     - Wait, let's re-read: \"Next year, the second sister will be twice as old as the first sister.\"\n       - Next year, the first sister will be 5 (since she's 4 now).\n       - So next year, the second sister will be twice as old as the first sister will be next year, i.e., twice 5 = 10 years old next year.\n       - Therefore, currently, the second sister is 10 - 1 = 9 years old.\n\n   Wait, let's double-check that interpretation because it seems a bit off. The wording is: \"Next year, the second sister will be twice as old as the first sister.\"\n\n   There are two ways to interpret this:\n   a) Next year, the second sister's age will be twice the first sister's age next year.\n   b) Next year, the second sister's age will be twice the first sister's current age.\n\n   The more natural interpretation is a): \"Next year, the second sister will be twice as old as the first sister [will be next year].\"\n\n   So:\n   - First sister's current age: 4\n   - Next year, first sister will be 5.\n   - Next year, second sister will be twice that, so 10.\n   - Therefore, second sister's current age is 10 - 1 = 9.\n\n   But wait, if next year second sister is 10, and first sister will be 5, then 10 is indeed twice 5. So current ages:\n   - First sister: 4\n   - Second sister: 9 (since next year she'll be 10)\n\n   But let's see if this makes sense with the rest of the problem.\n\n3. Third sister's age:\n   - The third sister is two years older than the second sister.\n   - Second sister is currently 9, so third sister is 9 + 2 = 11 years old.\n\n4. Older brother's age (John):\n   - The third sister is half the age of her older brother.\n   - Third sister is 11, so older brother is 11 * 2 = 22 years old.\n\nWait a minute, but the problem says there are 4 children: John and three sisters. So John is one of the four, and the other three are sisters. So if John is the older brother, then the four children are:\n1. First sister: 4\n2. Second sister: 9\n3. Third sister: 11\n4. John: ?\n\nBut the problem says \"the third sister is half the age of her older brother.\" So John is the older brother, meaning he's older than the third sister (11). So if third sister is half his age, then John is 22.\n\nBut let's check if this makes sense with all given information:\n- First sister: 4\n- Second sister: next year she'll be twice as old as the first sister will be next year.\n   - Next year, first sister: 5\n   - So second sister next year: 10, so now she's 9.\n- Third sister is two years older than second sister: 9 + 2 = 11.\n- Third sister is half as old as her older brother (John): so John is 22.\n\nNow, let's check the count of children:\n- John (brother)\n- First sister\n- Second sister\n- Third sister\nThat's 4 children, which matches the problem statement.\n\nBut wait, the problem says \"John is one of 4 children.\" So the four children are:\n1. John\n2. First sister\n3. Second sister\n4. Third sister\n\nBut the phrasing is: \"the third sister is half the age of her older brother.\" If John is the older brother, then he's older than the third sister (11), so 22 makes sense.\n\nBut let me think if there's another way to interpret it. Could \"her older brother\" refer to someone else? The problem says \"John is one of 4 children,\" and mentions three sisters, implying John is the brother. So likely the four are John and three sisters, making John the only brother.\n\nBut wait, the problem says \"the first sister is 4 years old,\" implying there are at least two sisters (since there's a first and a second). But it says \"the first sister,\" \"the second sister,\" and \"the third sister,\" implying three sisters and John, totaling four children.\n\nSo the ages are:\n- First sister: 4\n- Second sister: 9 (since next year she'll be 10, which is twice the first sister's age next year, which is 5)\n- Third sister: 11 (which is 9 + 2)\n- John: ?\n\nThe third sister is half the age of her older brother. So if the older brother is John, then John is 22.\n\nBut let's see if there's another possibility. Could there be another brother? The problem says John is one of 4 children, and mentions three sisters, implying John is the only brother. So likely John is the older brother referred to.\n\nBut let's think about the ages to see if this makes sense:\n- First sister: 4\n- Second sister: 9\n- Third sister: 11\n- John: 22\n\nIs this possible? Let's see the age gaps:\n- Between first and second sister: 9 - 4 = 5 years apart\n- Between second and third sister: 11 - 9 = 2 years apart (which matches \"the third sister is two years older than the second sister\")\n- Between third sister and John: 22 - 11 = 11 years apart\n\nIs there any inconsistency? The problem doesn't specify the age gaps between the sisters beyond what's given, so this seems plausible.\n\nBut let me think if there's another way to parse the problem. Maybe \"the first sister is 4 years old\" implies that the sisters are ordered by age, not necessarily by birth order. But usually, \"first sister\" would imply the oldest sister, but here the first sister is the youngest (4 years old), which is a bit odd unless \"first\" refers to something else, like order of mention. But likely it's by age, with first sister being the youngest.\n\nWait, if \"first sister\" is the youngest, then the order is first (youngest), second, third, and John is older than the third sister.\n\nIn the first interpretation:\n- First sister: 4\n- Second sister: 9\n- Third sister: 11\n- John: 22\n\nIn the second interpretation:\n- First sister: 4\n- Second sister: 7\n- Third sister: 9\n- John: 18\n\nBoth are possible based on the ambiguity in the problem statement. However, the more natural interpretation of the phrase \"next year, X will be twice as old as Y\" is that both ages are considered at the same future time (next year). So the first interpretation is more likely correct, making John 22.\n\nBut let me think if there's a way to confirm. The phrase is: \"Next year, the second sister will be twice as old as the first sister.\"\n\nIn mathematical terms:\nLet S1 be first sister's current age = 4\nLet S2 be second sister's current age.\nThen: S2 + 1 = 2 * (S1 + 1)\nS2 + 1 = 2 * (4 + 1) = 2 * 5 = 10\nSo S2 = 9.\n\nThen third sister is S3 = S2 + 2 = 11.\nAnd John's age J is such that S3 = J / 2 → J = 2 * S3 = 22.\n\nSo John is 22 years old.\n\nBut let me think if the ordering makes sense. If the sisters are ordered from first to third, and first is 4, second is 9, third is 11, and John is 22. That seems plausible, with John being the oldest.\n\nAlternatively, if \"first sister\" means the oldest sister, then the ages would be decreasing, but that would be unusual because we have the first sister as 4, which is very young, and then older sisters would be younger, which doesn't make sense. So likely \"first sister\" is the youngest, and they are ordered from youngest to oldest among the sisters, with John being older than all sisters.\n\nThus, John is 22 years old.\n\nBut wait, let's think about the numbering of sisters. Typically, \"first sister\" would refer to the firstborn sister, i.e., the oldest sister. But here, the first sister is 4, which would make her the youngest if she's the firstborn, which doesn't make sense because usually the firstborn is older. Maybe \"first sister\" means the first one mentioned, not necessarily by age.\n\nBut in English, when we say \"the first sister,\" it usually refers to the order in which they are listed or mentioned, not necessarily by age. But in the context of ages, it's more natural to assume that \"first sister\" is the oldest sister. But then the first sister is 4, which would make her the youngest if she's the firstborn, which is confusing.\n\nPerhaps \"first sister\" just means the first one we're mentioning, not necessarily the oldest. Similarly for second and third sisters. Then the order in which they are mentioned doesn't correspond to their ages. Then our earlier interpretation holds: first sister is 4, second is 9, third is 11, and John is 22.\n\nAlternatively, if \"first sister\" is the oldest sister, then:\n- first sister (oldest sister): 4\nWait, that can't be right because then the other sisters would be younger, but the second sister is older than the first (since next year she'll be twice as old as the first sister will be next year). If first sister is oldest at 4, then second sister would be younger, but next year she'd be older than the first sister next year, which contradicts the first sister being older. So \"first sister\" must not be the oldest; perhaps it's just an ordinal label not related to age.\n\nGiven that, and the more natural interpretation of the age comparison, John is likely 22.\n\nBut let me think if there's another way to interpret the family structure. The problem says John is one of 4 children. The other three are sisters. So the four children are: John, sister1, sister2, sister3.\n\nThe wording is: \"the first sister is 4 years old. Next year, the second sister will be twice as old as the first sister. The third sister is two years older than the second sister. The third sister is half the age of her older brother.\"\n\nSo the four children are:\n1. John (brother)\n2. First sister: 4\n3. Second sister: ?\n4. Third sister: ?\n\nAnd we know:\n- Next year, second sister's age = 2 * (first sister's age next year)\n  - First sister next year: 5\n  - So second sister next year: 10 → now 9\n- Third sister is two years older than second sister: 9 + 2 = 11\n- Third sister is half as old as her older brother (John): so John is 22.\n\nThis seems correct.\n\nBut in the alternative interpretation:\n- Next year, second sister's age = 2 * (first sister's current age) = 2*4 = 8 → now 7\n- Third sister: 7 + 2 = 9\n- John: 2 * 9 = 18\n\nBut in this case, who is the older brother? The third sister is 9, and John is 18, which fits (she is half his age). But then the sisters are 4, 7, and 9, and John is 18. That would mean the order from youngest to oldest is: first sister (4), second sister (7), third sister (9), and John (18). That seems plausible in terms of age gaps.\n\nBut in the first interpretation, the order would be:\nfirst sister: 4\nsecond sister: 9\nthird sister: 11\nJohn: 22\n\nThis also makes sense in terms of ordering, with ages increasing as we go from first to third sister, then John.\n\nBut in the first interpretation, the gap between third sister (11) and John (22) is 11 years, which is large but possible. In the second interpretation, the gap is 9 years (between 9 and 18), which is also large but slightly less so.\n\nGiven that both interpretations lead to consistent age orders and the problem's phrasing is ambiguous, but the first interpretation is more standard in math word problems, I think the correct answer is that John is 22 years old.\n\nHowever, considering that the second interpretation leads to more even age gaps and might be what the problem intended, it's possible that John is 18.\n\nGiven that in such problems, the more compact interpretation is often intended, and considering the more realistic age gaps in the second interpretation, perhaps the intended answer is 18.\n\nBut strictly based on the more precise grammatical interpretation, John is 22.\n\nGiven the ambiguity, and considering that the first interpretation is more aligned with standard problem phrasing, the most likely correct answer is that John is 22 years old.\n\nBut to resolve this, perhaps the problem expects us to interpret \"as the first sister\" to refer to her current age, leading to John being 18.\n\nGiven that, and considering that the second interpretation's age gaps seem more realistic, and that sometimes such problems expect the more compact interpretation, it's possible that the intended answer is 18.\n\nHowever, based on precise interpretation of the phrase, John is 22.\n\nGiven that both interpretations are possible, but the first is more grammatically precise, and considering that age gaps can vary widely in families, the most accurate answer based on the problem's wording is that John is 22 years old.\n\nBut looking back at the problem statement again:\n\n\"John is one of 4 children. The first sister is 4 years old. Next year, the second sister will be twice as old as the first sister. The third sister is two years older than the second sister. The third sister is half the age of her older brother. How old is John?\"\n\nThe phrase is: \"Next year, the second sister will be twice as old as the first sister.\"\n\nIf we interpret this as twice the first sister's age next year, then as calculated, John is 22.\n\nBut if we interpret it as twice the first sister's current age, then John is 18.\n\nGiven that the more natural interpretation in such contexts is the first one (comparing ages at the same future time), and considering that in mathematical problems, such phrases are typically interpreted that way, the correct answer is that John is 22 years old.\n\nBut let me see if there's a way to confirm. Suppose we take the first interpretation:\n\nCurrent ages:\n- First sister: 4\n- Second sister: 9 (since next year she'll be 10, which is twice the first sister's age next year, which is 5)\n- Third sister: 11\n- John: 22\n\nCheck:\n1. First sister is 4: correct.\n2. Next year, first sister will be 5, second sister will be 10, which is twice 5: correct.\n3. Third sister is 11, which is 2 more than second sister's 9: correct.\n4. Third sister (11) is half of John's age (22): correct.\n\nNow, second interpretation:\n\nCurrent ages:\n- First sister: 4\n- Second sister: 7 (since next year she'll be 8, which is twice the first sister's current age of 4)\nWait, no: if next year second sister is twice the first sister's current age (4), so next year second sister is 8, hence now she's 7.\n- Third sister: 9 (which is 7 + 2)\n- John: 18 (since third sister is half his age)\n\nCheck:\n1. First sister is 4: correct.\n2. Next year, second sister is 8, which is twice the first sister's current age (4): correct.\n3. Third sister is 9, which is 2 more than second sister's 7: correct.\n4. Third sister (9) is half of John's age (18): correct.\n\nBoth interpretations satisfy all given conditions. Therefore, the problem is ambiguous due to the phrasing of the second sentence. The more natural interpretation in mathematical contexts is the first one, where both ages being compared are at the same future time (next year), leading to John being 22.\n\nHowever, in everyday language, the phrase could be interpreted the second way, leading to John being 18.\n\nGiven that, and considering that the problem might expect the second interpretation, it's possible that the intended answer is 18. But based on precise interpretation, John is 22.\n\nGiven the ambiguity, and considering that in mathematics problems such comparisons are typically done at the same point in time, the more accurate answer is that John is 22 years old.\n\nBut to present both possibilities:\n\nIf \"next year, the second sister will be twice as old as the first sister will be at that time (next year),\" then John is 22.\n\nIf \"next year, the second sister will be twice as old as the first sister is now,\" then John is 18.\n\nGiven that the first interpretation is more standard in such problems, and considering that the age gaps in the second interpretation are more realistic but not definitive, the most likely correct answer is that John is 22 years old.\n\nHowever, given that both interpretations are possible due to the ambiguity in the problem's phrasing, and considering that the second interpretation leads to more plausible age gaps, it's also reasonable to conclude that John is 18.\n\nGiven that in such problems, the more compact interpretation is often intended, and considering that the second interpretation leads to more realistic age gaps, the most likely intended answer is that John is 18 years old.\n\nBut strictly based on precise interpretation of the phrase, John is 22.\n\nGiven the ambiguity, and since both interpretations satisfy all given conditions, the problem might have an intended interpretation leading to one of these answers. Given that the second interpretation leads to more realistic age gaps, and considering that sometimes such problems expect the more compact interpretation, the most likely intended answer is that John is 18 years old.\n\nTherefore, after careful consideration, the most plausible answer is that John is 18 years old.\n\nBut based on precise interpretation of the phrase, John is 22. Given that, and considering that in mathematics, such comparisons are typically done at the same point in time, the correct answer is that John is 22 years old.\n\nHowever, given the ambiguity in the phrasing, and considering that both interpretations are possible, and that the second interpretation leads to more realistic age gaps, it's also reasonable to conclude that John is 18.\n\nGiven that, and since both interpretations are valid based on the phrasing, but the first interpretation is more aligned with standard mathematical problem phrasing, the answer is that John is 22 years old.\n\nBut looking at similar problems, often the more compact interpretation is intended, so perhaps the answer is 18.\n\nGiven that both interpretations are possible, and considering that the second interpretation leads to more realistic age gaps, and that sometimes such problems expect the more compact interpretation, the most likely intended answer is that John is 18 years old.\n\nBut strictly based on precise interpretation, John is 22.\n\nGiven that, and considering that the problem's phrasing is ambiguous, but the more precise interpretation leads to John being 22, the correct answer is that John is 22 years old.\n\nHowever, in many such problems, the more compact interpretation is intended, so it's possible that the answer is 18.\n\nGiven that both interpretations satisfy all given conditions, and that the problem's phrasing is ambiguous, but the first interpretation is more precise, the answer is that John is 22 years old.\n\nBut to cover both possibilities, the problem is ambiguous, and John could be either 18 or 22 based on the interpretation.\n\nGiven that, and considering that the more precise interpretation leads to John being 22, and that's the answer that aligns with standard mathematical phrasing, the final answer is:"
              }
            ]
          },
          {
            "type": "text",
            "text": "To solve the problem, we start by analyzing the given information and the relationships between the ages of the sisters and John.\n\n1. The first sister is currently 4 years old.\n2. Next year, the second sister will be twice as old as the first sister will be at that time.\n3. The third sister is two years older than the second sister.\n4. The third sister is half the age of her older brother, John.\n\nLet's denote the current ages of the sisters and John as follows:\n- \\( S_1 \\): Age of the first sister\n- \\( S_2 \\): Age of the second sister\n- \\( S_3 \\): Age of the third sister\n- \\( J \\): Age of John\n\nFrom the given information:\n- \\( S_1 = 4 \\)\n- Next year, the second sister will be twice as old as the first sister will be next year. Therefore, next year, the first sister will be \\( 4 + 1 = 5 \\) years old, and the second sister will be \\( 2 \\times 5 = 10 \\) years old. This means the current age of the second sister is \\( 10 - 1 = 9 \\) years old.\n- The third sister is two years older than the second sister, so \\( S_3 = 9 + 2 = 11 \\) years old.\n- The third sister is half the age of her older brother, John. Therefore, \\( J = 2 \\times 11 = 22 \\) years old.\n\nThus, the age of John is \\(\\boxed{22}\\)."
          }
        ]
      }
    }
  ]
}

<think>\n*Thoughts and reasoning traces will go here.*\n</think>\n*Final answer will go here.*

<think>\nOkay, let's tackle this problem step by step. We have information about John and his three sisters. Let's list out what we know and try to derive each person's age.\n\n1. **First Sister**: The first sister is 4 years old. That's straightforward.\n\n   - First sister's age = 4 years\n\n2. **Second Sister**: Next year, the second sister will be twice as old as the first sister.\n\n   - First sister's age next year = 4 + 1 = 5 years\n   - So, second sister's age next year = 2 * 5 = 10 years\n   - Therefore, second sister's current age = 10 - 1 = 9 years\n\n3. **Third Sister**: The third sister is two years older than the second sister.\n\n   - Second sister's current age = 9 years\n   - So, third sister's age = 9 + 2 = 11 years\n\n4. **John (the older brother)**: The third sister is half the age of her older brother.\n\n   - Third sister's age = 11 years\n   - So, older brother's age = 11 * 2 = 22 years\n   - Therefore, John's age = 22 years\n\nBut wait, let's double-check this because the problem states that John is one of 4 children. That means there are four children in total: John and his three sisters. So, if John is the older brother, then the third sister is 11, and John is 22. But let's verify all the steps to ensure no mistakes were made.\n\n- First sister: 4 years old\n- Next year, first sister will be 5 years old\n- Next year, second sister will be twice as old as the first sister, so 2 * 5 = 10 years old next year\n- Therefore, second sister's current age is 10 - 1 = 9 years old\n- Third sister is two years older than the second sister: 9 + 2 = 11 years old\n- Third sister is half the age of her older brother: so older brother is 11 * 2 = 22 years old\n\nBut wait, if John is one of 4 children, and we have three sisters and John, that makes sense. The ages are:\n- First sister: 4\n- Second sister: 9\n- Third sister: 11\n- John: 22\n\nLet me check if all conditions are satisfied:\n\n1. First sister is 4 years old. ✔️\n2. Next year, first sister will be 5, and second sister will be 10, which is twice 5. ✔️\n3. Third sister is 11, which is 2 years older than the second sister (9 + 2 = 11). ✔️\n4. Third sister is half the age of her older brother. So older brother is 22, and half of that is 11, which matches the third sister's age. ✔️\n\nEverything checks out!\n\nBut wait, the problem says \"John is one of 4 children.\" That could mean that there are 4 children in total, including John. But in our count, we have three sisters and John, which is 4 children. So our initial assumption seems correct.\n\nBut let's think again: if John is one of 4 children, that means there are 3 other children besides John. So the three sisters and John would be 4 children. So our count seems correct.\n\nBut let's think about the wording: \"The third sister is half the age of her older brother.\" If John is the older brother, then that would mean John is older than the third sister. In our calculation, John is 22, and the third sister is 11, which fits because 11 is half of 22.\n\nBut just to be sure, let's consider if there's another older brother. The problem says \"the third sister is half the age of her older brother.\" It doesn't specify that John is the older brother, but since John is one of the 4 children, and we have three sisters, it's possible that John is the older brother.\n\nBut if there were another brother, then we'd have more than 4 children (three sisters, John, and another brother), which contradicts the given information that John is one of 4 children. So, John must be the older brother.\n\nTherefore, John's age is 22 years.\n\nWait a minute, let me re-examine the problem statement: \"John is one of 4 children. The first sister is 4 years old. Next year, the second sister will be twice as old as the first sister. The third sister is two years older than the second sister. The third sister is half the ago of her older brother. How old is John?\"\n\nOkay, so we have:\n\n1. Total children: 4 (John and 3 sisters)\n2. First sister: 4 years old\n3. Next year, second sister's age = 2 * (first sister's age next year)\n   - First sister next year: 4 + 1 = 5\n   - So, second sister next year: 2 * 5 = 10\n   - Therefore, second sister now: 10 - 1 = 9\n4. Third sister is two years older than the second sister: 9 + 2 = 11\n5. Third sister is half the age of her older brother: so older brother's age = 2 * 11 = 22\n\nNow, who is the older brother? The children are: John and three sisters. So the older brother must be John, unless there's another brother, but since there are only 4 children and three are sisters, John must be the brother.\n\nBut wait, the problem says \"the third sister is half the age of her older brother.\" If John is the brother, then he is the older brother, and his age is 22.\n\nBut let's think about the order of the sisters. The first sister is 4, second is 9, third is 11. So the ages are 4, 9, 11 for the sisters, and John is 22. That seems consistent because 22 is older than 11 (the third sister).\n\nBut let's check if there could be another brother. The problem says \"John is one of 4 children,\" and we have three sisters, so the fourth child must be John. Therefore, John is the only brother, and he is the older brother referred to in the problem.\n\nTherefore, John is 22 years old.\n\nBut wait, let's think about the phrasing: \"the third sister is half the age of her older brother.\" This implies that the older brother is not necessarily John. Maybe there's another brother older than John, making John not the oldest. But the problem states that John is one of 4 children, meaning the siblings are: three sisters and John, totaling 4 children. Therefore, there cannot be another brother; John must be the only brother and thus the older brother referred to.\n\nWait, but the problem says \"her older brother,\" implying that the older brother is someone other than the third sister. But since there are only four children: three sisters and John, and the third sister has an older brother, that older brother must be John.\n\nBut if John is the older brother, and the third sister is 11, then John would be 22. But is John necessarily the oldest? Could there be another brother older than John? No, because there are only four children: three sisters and John. So John must be the older brother.\n\nWait, no, the problem says \"the third sister is half the age of her older brother.\" It doesn't say that John is the older brother; it just says that John is one of the four children. So the four children are: three sisters and John. And the third sister has an older brother, which must be John, because there are no other brothers. So John must be the older brother, making him 22.\n\nBut let me think again: the four children are: first sister, second sister, third sister, and John. The third sister has an older brother. Among the four children, the possible older brothers would be... well, the sisters are all girls, so the only possible brother is John. Therefore, John must be the older brother, and his age is 22.\n\nBut wait, could one of the sisters be older than John? The problem doesn't specify the order of ages among the siblings, only that the third sister has an older brother. But since all the other siblings are sisters (three sisters and John), the older brother must be John.\n\nWait, no, the siblings are: first sister, second sister, third sister, and John. The third sister is 11. Her older brother would have to be older than 11. The other sisters are 4 and 9, so they are younger. Therefore, the older brother must be John, and his age is 22.\n\nBut let me think if there's another interpretation. Maybe \"the third sister is half the age of her older brother\" implies that the older brother is not necessarily John. But since there are only four children (three sisters and John), the older brother must be John.\n\nWait, unless one of the sisters is older than John, but then that sister would be the older sister, not brother. So the older brother must be John.\n\nTherefore, John is 22 years old.\n\nBut let me think again about the siblings' ages to ensure consistency:\n\n- First sister: 4\n- Second sister: 9\n- Third sister: 11\n- John: ?\n\nFrom the problem: \"The third sister is half the age of her older brother.\" So older brother's age = 2 * third sister's age = 2 * 11 = 22. So John is 22.\n\nBut let's see the order of ages: 4, 9, 11, and 22. That seems consistent because 22 is older than 11, so John is indeed the older brother.\n\nWait a minute, but the first sister is 4, second is 9, third is 11, and John is 22. That would mean the order from youngest to oldest is: first sister (4), second sister (9), third sister (11), and John (22). That makes sense because each subsequent sibling is older than the previous one, and John is the oldest.\n\nTherefore, John is 22 years old.\n\nBut before finalizing, let me check if there's another way to interpret the problem. Maybe the order of the sisters is not by age but just by mention. The problem says \"the first sister,\" not necessarily the youngest. But usually, when we say first, second, third, it's in order of age, with first being the youngest, but that's not necessarily the case. However, in age problems, it's common to list siblings in order of age, but the problem doesn't specify.\n\nBut the problem says \"the first sister is 4 years old,\" and then talks about the second and third sisters, implying that first, second, third refer to their order in the family, not necessarily by age. But usually, first, second, third would refer to birth order, meaning first is the oldest, but that contradicts the ages we have because the first sister is 4, which would be the youngest.\n\nWait, in English, when we say \"first sister,\" \"second sister,\" etc., it typically refers to the order of birth, with first being the oldest. But in this case, the first sister is 4 years old, which is younger than the second sister (9) and third sister (11). So \"first,\" \"second,\" \"third\" here must refer to some other order, not birth order.\n\nBut in common usage, first, second, third sister would refer to birth order, with first being the oldest. If that's the case, then:\n\n- First sister (oldest sister): 4 years old\n- Second sister: ?\n- Third sister: ?\nBut then the first sister being the oldest at 4 doesn't make sense because the other sisters would have to be younger, but the second sister is 9 next year which would make her older than 4 now, which contradicts the first sister being the oldest.\n\nTherefore, \"first,\" \"second,\" \"third\" must not refer to birth order but perhaps the order in which they are mentioned or some other order. For the sake of the problem, it's likely that \"first,\" \"second,\" \"third\" are just labels and not indicative of birth order, because if first sister is 4 and second is older (9), that's not possible if first is the oldest.\n\nSo let's assume that \"first,\" \"second,\" \"third\" are just labels and not indicative of birth order.\n\nThen we have:\n\n- Sister A (first sister): 4\n- Sister B (second sister): current age is 9 (since next year she'll be 10)\n- Sister C (third sister): 11\n- John: ?\n\nFrom the problem: \"The third sister is half the age of her older brother.\" So older brother's age is 22, hence John is 22.\n\nBut is John necessarily the older brother? There are four children: three sisters and John. So the siblings are: Sister A (4), Sister B (9), Sister C (11), and John.\n\nFor Sister C (11) to have an older brother, that brother must be older than 11. The other siblings are 4, 9, and John. So John must be older than 11 to be the older brother. So John is 22, which is older than 11, so that checks out.\n\nBut wait, could one of the sisters be older than John? The sisters are 4, 9, and 11. So none of the sisters are older than John if John is 22.\n\nBut is John necessarily the older brother? The problem says John is one of the four children, and the four children are three sisters and John. So the older brother must be John, because the other siblings are sisters.\n\nTherefore, John is the older brother, and his age is 22.\n\nBut let me think if there's any ambiguity. The problem says \"the third sister is half the age of her older brother.\" It doesn't specify that the older brother is John; it just says \"her older brother.\" But among the four children (three sisters and John), the only possible older brother is John, because the other siblings are sisters. So John must be the older brother, making him 22.\n\nBut let me think if there's any other possibility. Could \"her older brother\" refer to someone outside the four children? No, because the four children consist of John and his three sisters, so the older brother must be within these four, which is John.\n\nTherefore, John is 22 years old.\n\nBut let me think one more time to ensure no mistakes. Let's list the siblings and their ages:\n\n1. First sister: 4 years old\n2. Second sister: currently 9 (since next year she'll be 10, which is twice the first sister's age next year, which is 5)\n3. Third sister: 11 (which is 2 years older than the second sister's current age of 9)\n4. John: ?\n\nThe third sister is half the age of her older brother. So older brother's age = 2 * 11 = 22. Therefore, John is 22.\n\nNow, is John indeed the older brother? Let's see the ages: 4, 9, 11, and 22. So the order from youngest to oldest is: 4, 9, 11, 22. So John at 22 is the oldest, hence the older brother to the third sister (11).\n\nTherefore, John is 22 years old.\n\nBut wait, the problem says \"John is one of 4 children.\" So the four children are: first sister (4), second sister (9), third sister (11), and John (22). That adds up to four children, and John is indeed the older brother to the third sister.\n\nBut let me think if there's another way to interpret the problem. Maybe \"the first sister is 4 years old\" doesn't necessarily mean she's the youngest. Maybe the sisters are ordered differently. But the problem lists them as first, second, third, which typically implies birth order where first is the oldest. But if first sister is 4, and second is older (since next year she's twice as old as first sister next year), that contradicts first being the oldest.\n\nWait, let's assume \"first,\" \"second,\" \"third\" refer to birth order, where first is the oldest.\n\nThen:\n\n- First sister (oldest): 4\nBut then the second sister is younger than the first, but next year the second sister will be twice as old as the first sister next year.\n\nLet’s try this interpretation.\n\n1. First sister (oldest): 4 years old\n2. Next year:\n   - First sister: 4 + 1 = 5\n   - Second sister will be twice as old as first sister next year: so second sister next year will be 2 * 5 = 10, so current age is 10 - 1 = 9\nBut if first sister is oldest at 4, second sister is younger, so should be less than 4, but we got 9, which is older than 4. This contradicts the assumption that first sister is the oldest.\n\nTherefore, \"first,\" \"second,\" \"third\" cannot refer to birth order where first is oldest. Maybe \"first,\" \"second,\" \"third\" are just labels not indicating age order.\n\nSo let's stick with the initial interpretation where \"first,\" \"second,\" \"third\" are just labels, not indicating birth order, and we have:\n\n- First sister: 4\n- Second sister: currently 9 (since next year she'll be 10, which is twice the first sister's age next year, which is 5)\n- Third sister: 11 (which is 2 years older than the second sister's current age of 9)\n- John: ?\n\nThird sister is half the age of her older brother, so older brother is 22, hence John is 22.\n\nThis seems consistent. Let me list the siblings and their ages:\n\n1. First sister: 4\n2. Second sister: 9\n3. Third sister: 11\n4. John: 22\n\nNow, let's check the birth order based on ages. The order from youngest to oldest would be: 4 (first sister), 9 (second sister), 11 (third sister), 22 (John).\n\nSo in terms of birth order (assuming older siblings are born first), the order would be:\n\n1. John: 22 (oldest)\n2. Third sister: 11\n3. Second sister: 9\n4. First sister: 4 (youngest)\n\nBut the problem refers to them as first, second, third sisters, not in birth order but perhaps in some other order (maybe order of mention in the problem).\n\nGiven that, and since John is the oldest (22), then the third sister (11) has an older brother, which is John, satisfying the condition.\n\nTherefore, John is 22 years old.\n\n</think>\n\nJohn is 22 years old.\n\nHere's the summary:\n\nWe are given that John is one of four children, with three sisters. The first sister is 4 years old. Next year, the second sister will be twice as old as the first sister will be next year (which is 5), making the second sister currently 9 years old. The third sister is two years older than the second sister, so she is 11 years old. The third sister is half the age of her older brother, which means the older brother is 22 years old. Since John is the only brother among the four children (three sisters and John), he must be the older brother, making John 22 years old.\n\nTherefore, John is 22 years old.
```
</details>

We can see that the model is able to reason through the problem and arrive at the correct answer, let's format each section in markdown:

<details>
<summary><b>Rendered Thinking</b></summary>

Okay, let's tackle this problem step by step. We have information about John and his three sisters. Let's list out what we know and try to derive each person's age.

1. **First Sister**: The first sister is 4 years old. That's straightforward.

- First sister's age = 4 years

2. **Second Sister**: Next year, the second sister will be twice as old as the first sister.

- First sister's age next year = 4 + 1 = 5 years
   - So, second sister's age next year = 2 * 5 = 10 years
   - Therefore, second sister's current age = 10 - 1 = 9 years

3. **Third Sister**: The third sister is two years older than the second sister.

- Second sister's current age = 9 years
   - So, third sister's age = 9 + 2 = 11 years

4. **John (the older brother)**: The third sister is half the age of her older brother.

- Third sister's age = 11 years
   - So, older brother's age = 11 * 2 = 22 years
   - Therefore, John's age = 22 years

But wait, let's double-check this because the problem states that John is one of 4 children. That means there are four children in total: John and his three sisters. So, if John is the older brother, then the third sister is 11, and John is 22. But let's verify all the steps to ensure no mistakes were made.

- First sister: 4 years old
- Next year, first sister will be 5 years old
- Next year, second sister will be twice as old as the first sister, so 2 * 5 = 10 years old next year
- Therefore, second sister's current age is 10 - 1 = 9 years old
- Third sister is two years older than the second sister: 9 + 2 = 11 years old
- Third sister is half the age of her older brother: so older brother is 11 * 2 = 22 years old

But wait, if John is one of 4 children, and we have three sisters and John, that makes sense. The ages are:
- First sister: 4
- Second sister: 9
- Third sister: 11
- John: 22

Let me check if all conditions are satisfied:

1. First sister is 4 years old. ✔️
2. Next year, first sister will be 5, and second sister will be 10, which is twice 5. ✔️
3. Third sister is 11, which is 2 years older than the second sister (9 + 2 = 11). ✔️
4. Third sister is half the age of her older brother. So older brother is 22, and half of that is 11, which matches the third sister's age. ✔️

Everything checks out!

But wait, the problem says "John is one of 4 children." That could mean that there are 4 children in total, including John. But in our count, we have three sisters and John, which is 4 children. So our initial assumption seems correct.

But let's think again: if John is one of 4 children, that means there are 3 other children besides John. So the three sisters and John would be 4 children. So our count seems correct.

But let's think about the wording: "The third sister is half the age of her older brother." If John is the older brother, then that would mean John is older than the third sister. In our calculation, John is 22, and the third sister is 11, which fits because 11 is half of 22.

But just to be sure, let's consider if there's another older brother. The problem says "the third sister is half the age of her older brother." It doesn't specify that John is the older brother; it just says that John is one of the four children. So the four children are: three sisters and John. And the third sister has an older brother, which must be John, because there are no other brothers. So John must be the older brother, making him 22.

But if there were another brother, then we'd have more than 4 children (three sisters, John, and another brother), which contradicts the given information that John is one of 4 children. So, John must be the older brother.

Therefore, John's age is 22 years.

Wait a minute, let me re-examine the problem statement: "John is one of 4 children. The first sister is 4 years old. Next year, the second sister will be twice as old as the first sister. The third sister is two years older than the second sister. The third sister is half the ago of her older brother. How old is John?"

1. Total children: 4 (John and 3 sisters)
2. First sister: 4 years old
3. Next year, second sister's age = 2 * (first sister's age next year)
   - First sister next year: 4 + 1 = 5
   - So, second sister next year: 2 * 5 = 10
   - Therefore, second sister now: 10 - 1 = 9
4. Third sister is two years older than the second sister: 9 + 2 = 11
5. Third sister is half the age of her older brother: so older brother's age = 2 * 11 = 22

Now, who is the older brother? The children are: John and three sisters. So the older brother must be John, unless there's another brother, but since there are only 4 children and three are sisters, John must be the brother.

But wait, the problem says "the third sister is half the age of her older brother." If John is the brother, then he is the older brother, and his age is 22.

But let's think about the order of the sisters. The first sister is 4, second is 9, third is 11. So the ages are 4, 9, 11 for the sisters, and John is 22. That seems consistent because 22 is older than 11 (the third sister).

But let's check if there could be another brother. The problem says "John is one of 4 children," and we have three sisters, so the fourth child must be John. Therefore, John is the only brother, and he is the older brother referred to in the problem.

Therefore, John is 22 years old.

But wait, let's think about the phrasing: "the third sister is half the age of her older brother." This implies that the older brother is not necessarily John. Maybe there's another brother older than John, making John not the oldest. But the problem states that John is one of 4 children, meaning the siblings are: three sisters and John, totaling 4 children. Therefore, there cannot be another brother; John must be the only brother and thus the older brother referred to.

Wait, but the problem says "her older brother," implying that the older brother is someone other than the third sister. But since there are only four children: three sisters and John, and the third sister has an older brother, that older brother must be John.

But if John is the older brother, and the third sister is 11, then John would be 22. But is John necessarily the oldest? Could there be another brother older than John? No, because there are only four children: three sisters and John. So John must be the older brother.

Wait, no, the problem says "the third sister is half the age of her older brother." It doesn't say that John is the older brother; it just says that John is one of the four children. So the four children are: three sisters and John. And the third sister has an older brother, which must be John, because there are no other brothers. So John must be the older brother, making him 22.

But just to be sure, let's consider if there's another interpretation. Maybe "the first sister is 4 years old" doesn't necessarily mean she's the youngest. Maybe the sisters are ordered differently. But the problem lists them as first, second, third, which typically implies birth order, with first being the oldest. But in this case, the first sister is 4, which is younger than the second sister (9) and third sister (11). So "first," "second," "third" here must refer to some other order, not birth order.

But in common usage, first, second, third sister would refer to birth order, with first being the oldest. If that's the case, then:

- First sister (oldest sister): 4 years old
- Second sister: ?
- Third sister: ?

But then the first sister being the oldest at 4 doesn't make sense because the other sisters would have to be younger, but the second sister is 9 next year which would make her older than 4 now, which contradicts the first sister being the oldest.

Therefore, "first," "second," "third" must not refer to birth order but perhaps the order in which they are mentioned or some other order. For the sake of the problem, it's likely that "first," "second," "third" are just labels and not indicative of birth order, because if first sister is 4 and second is older (9), that's not possible if first is the oldest.

So let's assume that "first," "second," "third" are just labels and not indicative of birth order.

- Sister A (first sister): 4
- Sister B (second sister): current age is 9 (since next year she'll be 10)
- Sister C (third sister): 11
- John: ?

From the problem: "The third sister is half the age of her older brother." So older brother's age is 22, hence John is 22.

But is John necessarily the older brother? There are four children: three sisters and John. So the siblings are: Sister A (4), Sister B (9), Sister C (11), and John.

For Sister C (11) to have an older brother, that brother must be older than 11. The other siblings are 4, 9, and John. So John must be older than 11 to be the older brother. So John is 22, which is older than 11, so that checks out.

But wait, could one of the sisters be older than John? The sisters are 4, 9, and 11. So none of the sisters are older than John if John is 22.

But is John necessarily the older brother? The problem says John is one of the four children, and the four children are three sisters and John. So the older brother must be John, because the other siblings are sisters.

Therefore, John is the older brother, and his age is 22.

But let me think if there's any ambiguity. The problem says "the third sister is half the age of her older brother." It doesn't specify that the older brother is John; it just says "her older brother." But among the four children (three sisters and John), the only possible older brother is John, because the other siblings are sisters. So John must be the older brother, making him 22.

But let me think if there's any other possibility. Could "her older brother" refer to someone outside the four children? No, because the four children consist of John and his three sisters, so the older brother must be within these four, which is John.

Therefore, John is 22 years old.

But let me think one more time to ensure no mistakes. Let's list the siblings and their ages:

1. First sister: 4 years old
2. Second sister: currently 9 (since next year she'll be 10, which is twice the first sister's age next year, which is 5)
3. Third sister: 11 (which is 2 years older than the second sister's current age of 9)
4. John: ?

The third sister is half the age of her older brother. So older brother's age = 2 * 11 = 22. Therefore, John is 22.

Now, is John indeed the older brother? Let's see the ages: 4, 9, 11, and 22. So the order from youngest to oldest is: 4, 9, 11, 22. So John at 22 is the oldest, hence the older brother to the third sister (11).

Therefore, John is 22 years old.

But wait, the problem says "John is one of 4 children." So the four children are: first sister (4), second sister (9), third sister (11), and John (22). That adds up to four children, and John is indeed the older brother to the third sister.

But let me think if there's another way to interpret the problem. Maybe "the first sister is 4 years old" doesn't necessarily mean she's the youngest. Maybe the sisters are ordered differently. But the problem lists them as first, second, third, which typically implies birth order where first is the oldest. But if first sister is 4, and second is older (since next year she's twice as old as first sister next year), that contradicts first being the oldest.

Wait, let's assume "first," "second," "third" refer to birth order, where first is the oldest.

- First sister (oldest): 4
But then the second sister is younger than the first, but next year she's twice as old as first sister next year.

Let’s try this interpretation.

1. First sister (oldest): 4 years old
2. Next year:
   - First sister: 4 + 1 = 5
   - Second sister will be twice as old as first sister next year: so second sister next year will be 2 * 5 = 10, so current age is 10 - 1 = 9
But if first sister is oldest at 4, second sister is younger, so should be less than 4, but we got 9, which is older than 4. This contradicts the assumption that first is the oldest.

Therefore, "first," "second," "third" cannot refer to birth order where first is oldest. Maybe "first," "second," "third" are just labels not indicating age order.

So let's stick with the initial interpretation where "first," "second," "third" are just labels, not indicating birth order, and we have:

- First sister: 4
- Second sister: currently 9 (since next year she'll be 10, which is twice the first sister's age next year, which is 5)
- Third sister: 11 (which is 2 years older than the second sister's current age of 9)
- John: ?

Third sister is half the age of her older brother, so older brother is 22, hence John is 22.

This seems consistent. Let me list the siblings and their ages:

1. First sister: 4
2. Second sister: 9
3. Third sister: 11
4. John: 22

Now, let's check the birth order based on ages. The order from youngest to oldest would be: 4 (first sister), 9 (second sister), 11 (third sister), 22 (John).

So in terms of birth order (assuming older siblings are born first), the order would be:

1. John: 22 (oldest)
2. Third sister: 11
3. Second sister: 9
4. First sister: 4 (youngest)

But the problem refers to them as first, second, third sisters, not in birth order but perhaps in some other order (maybe order of mention in the problem).

Given that, and since John is the oldest (22), then the third sister (11) has an older brother, which is John, satisfying the condition.

Therefore, John is 22 years old.
</details>

<details>
<summary><b>Rendered Answer</b></summary>

John is 22 years old.

We are given that John is one of four children, with three sisters. The first sister is 4 years old. Next year, the second sister will be twice as old as the first sister will be next year (which is 5), making the second sister currently 9 years old. The third sister is two years older than the second sister, so she is 11 years old. The third sister is half the age of her older brother, which means the older brother is 22 years old. Since John is the only brother among the four children (three sisters and John), he must be the older brother, making John 22 years old.

Therefore, John is 22 years old.
</details>

[Custom Structured Output]
Source: https://docs.mistral.ai/docs/capabilities/structured-output/custom

**Examples:**

Example 1 (unknown):
```unknown
</TabItem>
  <TabItem value="typescript" label="typescript">
```

Example 2 (unknown):
```unknown
</TabItem>
</Tabs>

## Step 1. User: specify tools and query

<div style={{ textAlign: 'center' }}>
  <img
    src="/img/guides/functioncalling2.png"
    alt="functioncalling2"
    width="600"
    style={{ borderRadius: '15px' }}
  />
</div>


### Tools
Users can define all the necessary tools for their use cases. 

- In many cases, we might have multiple tools at our disposal. For example, let’s consider we have two functions as our two tools: `retrieve_payment_status` and `retrieve_payment_date` to retrieve payment status and payment date given transaction ID.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>
```

Example 3 (unknown):
```unknown
</TabItem>
  <TabItem value="typescript" label="typescript">
```

Example 4 (unknown):
```unknown
</TabItem>
</Tabs>

- In order for Mistral models to understand the functions, we need to outline the function specifications with a JSON schema. Specifically, we need to describe the type, function name, function description, function parameters, and the required parameter for the function.  Since we have two functions here, let’s list two function specifications in a list. 

<Tabs groupId="code">
  <TabItem value="python" label="python" default>
```

---

## Create or update an access level.

**URL:** llms-txt#create-or-update-an-access-level.

Source: https://docs.mistral.ai/api/#tag/libraries_share_create_v1

put /v1/libraries/{library_id}/share

---

## Custom Structured Outputs

**URL:** llms-txt#custom-structured-outputs

**Contents:**
  - Define the Data Model
  - Start the completion
  - Define the Data Model
  - Start the completion
  - FAQ

Custom Structured Outputs allow you to ensure the model provides an answer in a very specific JSON format by supplying a clear JSON schema. This approach allows the model to consistently deliver responses with the correct typing and keywords.

<Tabs>
  <TabItem value="python" label="python" default>

Here is an example of how to achieve this using the Mistral AI client and Pydantic:

### Define the Data Model

First, define the structure of the output using a Pydantic model:

### Start the completion

Next, use the Mistral AI python client to make a request and ensure the response adheres to the defined structure using `response_format` set to the corresponding pydantic model:

In this example, the `Book` class defines the structure of the output, ensuring that the model's response adheres to the specified format.

There are two types of possible outputs that are easily accessible via our SDK:

1. The raw JSON output, accessed with `chat_response.choices[0].message.content`:

2. The parsed output, converted into a Pydantic object with `chat_response.choices[0].message.parsed`. In this case, it is a `Book` instance:

<TabItem value="typescript" label="typescript">

Here is an example of how to achieve this using the Mistral AI client and Zod:

### Define the Data Model

First, define the structure of the output using Zod:

### Start the completion

Next, use the Mistral AI TypeScript client to make a request and ensure the response adheres to the defined structure using `responseFormat` set to the corresponding Zod schema:

In this example, the `Book` schema defines the structure of the output, ensuring that the model's response adheres to the specified format.

There are two types of possible outputs that are easily accessible via our SDK:

1. The raw JSON output, accessed with `chatResponse.choices[0].message.content`:

2. The parsed output, converted into a TypeScript object with `chatResponse.choices[0].message.parsed`. In this case, it is a `Book` object:

</TabItem>
  <TabItem value="curl" label="curl">

The request is structured to ensure that the response adheres to the specified custom JSON schema. The `schema` defines the structure of a Book object with name and authors properties.

:::note
To better guide the model, the following is being always prepended by default to the System Prompt when using this method:

However, it is recommended to add more explanations and iterate on your system prompt to better clarify the expected schema and behavior.
:::

### FAQ
**Q: Which models support custom Structured Outputs?**  
**A:** All currently available models except for `codestral-mamba` are supported.

[JSON mode]
Source: https://docs.mistral.ai/docs/capabilities/structured-output/json-mode

Users have the option to set `response_format` to `{"type": "json_object"}` to enable JSON mode.
Currently, JSON mode is available for all of our models through API.

<Tabs>
  <TabItem value="python" label="python" default>

</TabItem>
  <TabItem value="typescript" label="typescript">

</TabItem>
  <TabItem value="curl" label="curl">

[Structured Output]
Source: https://docs.mistral.ai/docs/capabilities/structured-output/overview

**Examples:**

Example 1 (python):
```python
from pydantic import BaseModel

class Book(BaseModel):
    name: str
    authors: list[str]
```

Example 2 (python):
```python
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "ministral-8b-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.parse(
    model=model,
    messages=[
        {
            "role": "system", 
            "content": "Extract the books information."
        },
        {
            "role": "user", 
            "content": "I recently read 'To Kill a Mockingbird' by Harper Lee."
        },
    ],
    response_format=Book,
    max_tokens=256,
    temperature=0
)
```

Example 3 (json):
```json
{
  "authors": ["Harper Lee"],
  "name": "To Kill a Mockingbird"
}
```

Example 4 (python):
```python
name='To Kill a Mockingbird' authors=['Harper Lee']
```

---

## defaultdict(list,

**URL:** llms-txt#defaultdict(list,

---

## define prompt template

**URL:** llms-txt#define-prompt-template

prompt_template = """Write a Python function to execute the following task: {task} 
Return only valid Python code. Do not give any explanation.
Never start with

- Step 2: Decide how we evaluate the code generation

Next, we can use `code_eval.compute` to evaluate whether the generated code passes the pre-defined unit tests. `predictions` is a list of generated code outputs from the LLM and `references` is a list of unit tests for each prediction.

:::warning[ ]
 This code is designed to run code that’s been generated by a model, which may not be entirely reliable. While it's unlikely that the model-generated code will intentionally cause harm, it's strongly recommended to run this evaluation suite in a sandbox environment, isolating it from your main system and network.
:::

```py
from evaluate import load

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
code_eval = load("code_eval")

**Examples:**

Example 1 (unknown):
```unknown
- Step 2: Decide how we evaluate the code generation

Next, we can use `code_eval.compute` to evaluate whether the generated code passes the pre-defined unit tests. `predictions` is a list of generated code outputs from the LLM and `references` is a list of unit tests for each prediction. 

:::warning[ ]
 This code is designed to run code that’s been generated by a model, which may not be entirely reliable. While it's unlikely that the model-generated code will intentionally cause harm, it's strongly recommended to run this evaluation suite in a sandbox environment, isolating it from your main system and network.
:::
```

---

## deleted_document = client.beta.libraries.documents.delete(library_id=new_library.id, document_id=uploaded_doc.id)

**URL:** llms-txt#deleted_document-=-client.beta.libraries.documents.delete(library_id=new_library.id,-document_id=uploaded_doc.id)

**Contents:**
  - Control Access
- Create a Document Library Agent
- How It Works
  - Conversations with Document Library
  - Explanation of the Outputs
- Create an Image Generation Agent
- How It Works
  - Conversations with Image Generation
  - Explanation of the Outputs
  - Download Images

typescript
// Get document info once processed
const deletedLibrary = await client.beta.libraries.delete({
    libraryId: newLibrary.id
});
// const deletedDocument = await client.beta.libraries.documents.delete({
//    libraryId: newLibrary.id,
//    documentId: uploadedDoc.id
// });
bash
curl --location --request DELETE "https://api.mistral.ai/v1/libraries/<library_id>" \
     --header "Accept: application/json" \
     --header "Authorization: Bearer $MISTRAL_API_KEY"
bash
curl --location --request DELETE "https://api.mistral.ai/v1/libraries/<library_id>/documents/<document_id>" \
     --header "Accept: application/json" \
     --header "Authorization: Bearer $MISTRAL_API_KEY"
json
{
  "id": "0197f425-5e85-7353-b8e7-e8b974b9c613",
  "name": "Mistral Models",
  "created_at": "2025-07-10T11:42:59.230268Z",
  "updated_at": "2025-07-10T12:05:59.638182Z",
  "owner_id": "6340e568-a546-4c41-9dee-1fbeb80493e1",
  "owner_type": "Workspace",
  "total_size": 3749788,
  "nb_documents": 1,
  "chunk_size": null,
  "emoji": null,
  "description": "A simple library with information about Mistral models.",
  "generated_name": null,
  "generated_description": "A library featuring Mistral 7B, a high-performing language model with advanced attention mechanisms.",
  "explicit_user_members_count": null,
  "explicit_workspace_members_count": null,
  "org_sharing_role": null
}
python
accesses_list = client.beta.libraries.accesses.list(library_id=new_library.id)
typescript
const accessesList = await client.beta.libraries.accesses.list({
    libraryId: newLibrary.id
});
bash
curl --location "https://api.mistral.ai/v1/libraries/<library_id>/share" \
     --header "Accept: application/json" \
     --header "Authorization: Bearer $MISTRAL_API_KEY"
python
access = client.beta.libraries.accesses.update_or_create(
    library_id=new_library.id,
    org_id="<org_id>",
    level="<level_type>",
    share_with_uuid="<uuid>",
    share_with_type="<account_type>"
)
typescript
const access = await client.beta.libraries.accesses.updateOrCreate({
    libraryId: newLibrary.id,
    sharingIn:{
        orgId: "<orgId>",
        level: "<levelType>",
        shareWithUuid: "<uuid>",
        shareWithType: "<accountType>"
    }
});
bash
curl --location --request PUT "https://api.mistral.ai/v1/libraries/<library_id>/share" \
     --header "Accept: application/json" \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --header "Content-Type: application/json" \
     --data '{
         "org_id": "<org_id>",
         "level": "<level_type>",
         "share_with_uuid": "<uuid>",
         "share_with_type": "<account_type>"
     }'
python
access_deleted = client.beta.libraries.accesses.delete(
    library_id=new_library.id,
    org_id="<org_id>",
    share_with_uuid="<uuid>",
    share_with_type="<account_type>"
)
typescript
const accessDeleted = await client.beta.libraries.accesses.delete({
    libraryId: newLibrary.id,
    sharingDelete: {
        orgId: "<orgId>",
        shareWithUuid: "<uuid>",
        shareWithType: "<accountType>"
    }
});
bash
curl --location --request DELETE "https://api.mistral.ai/v1/libraries/<library_id>/share" \
     --header "Accept: application/json" \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --header "Content-Type: application/json" \
     --data '{
         "org_id": "<org_id>",
         "share_with_uuid": "<uuid>",
         "share_with_type": "<account_type>"
     }'
py
library_agent = client.beta.agents.create(
    model="mistral-medium-2505",
    name="Document Library Agent",
    description="Agent used to access documents from the document library.",
    instructions="Use the  library tool to access external documents.",
    tools=[{"type": "document_library", "library_ids": [new_library.id]}],
    completion_args={
        "temperature": 0.3,
        "top_p": 0.95,
    }
)
typescript
let libraryAgent = await client.beta.agents.create({
    model:"mistral-medium-2505",
    name:"Document Library Agent",
    description:"Agent used to access documents from the document library.",
    instructions:"Use the  library tool to access external documents.",
    tools:[
        {
            type: "document_library", 
            libraryIds: [newLibrary.id]
        }
    ],
    completionArgs:{
        temperature: 0.3,
        topP: 0.95,
    }
});
bash
curl --location "https://api.mistral.ai/v1/agents" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
     "model": "mistral-medium-2505",
     "name": "Library Agent",
     "description": "Agent able to search information in your library...",
     "instructions": "You have the ability to perform searches with `document_library` to find relevant information.",
     "tools": [
       {
         "type": "document_library",
         "library_ids" : ["<library_id>"]
       }
     ],
     "completion_args": {
       "temperature": 0.3,
       "top_p": 0.95
     }
  }'
json
{
  "model": "mistral-medium-2505",
  "name": "Document Library Agent",
  "description": "Agent used to access documents from the document library.",
  "id": "ag_06835bb196f9720680004fb1873efbae",
  "version": 0,
  "created_at": "2025-05-27T13:16:09.438785Z",
  "updated_at": "2025-05-27T13:16:09.438787Z",
  "instructions": "Use the library tool to access external documents.",
  "tools": [
    {
      "library_ids": [
        "06835a9c-262c-7e83-8000-594d29fe2948"
      ],
      "type": "document_library"
    }
  ],
  "completion_args": {
    "stop": null,
    "presence_penalty": null,
    "frequency_penalty": null,
    "temperature": 0.3,
    "top_p": 0.95,
    "max_tokens": null,
    "random_seed": null,
    "prediction": null,
    "response_format": null,
    "tool_choice": "auto"
  },
  "handoffs": null,
  "object": "agent"
}

py
response = client.beta.conversations.start(
    agent_id=image_agent.id,
    inputs="How does the vision encoder for pixtral 12b work"
)
typescript
let conversation = await client.beta.conversations.start({
    agentId: libraryAgent.id,
    inputs: "How does the vision encoder for pixtral 12b work"
});
bash
curl --location "https://api.mistral.ai/v1/conversations" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
     "inputs": "How does the vision encoder for pixtral 12b work",
     "stream": false,
     "agent_id": "<agent_id>"
  }'
json
{
  "conversation_id": "conv_06835bb1996079898000435d8a0b1afd",
  "outputs": [
    {
      "type": "tool.execution",
      "name": "document_library",
      "object": "entry",
      "created_at": "2025-05-27T13:16:09.974925Z",
      "completed_at": "2025-05-27T13:16:10.855373Z",
      "id": "tool_exec_06835bb19f99716580001de8ab64d953"
    },
    {
      "type": "message.output",
      "content": [
        {
          "type": "text",
          "text": "The vision encoder for Pixtral 12B, known as PixtralViT, is designed to process images at their natural resolution and aspect ratio. Here are the key details about how it works:\n\n1. **Architecture**: PixtralViT is a vision transformer with 400 million parameters. It is trained from scratch to support variable image sizes and aspect ratios, which is a significant departure from standard architectures that often require fixed image sizes.\n\n2. **Key Modifications**:\n   - **Break Tokens**: To help the model distinguish between images with the same number of patches but different aspect ratios, special tokens like [IMAGE BREAK] are inserted between image rows, and an [IMAGE END] token is added at the end of an image sequence.\n   - **Gating in FFN**: Instead of using a standard feedforward layer in the attention block, PixtralViT employs gating in the hidden layer, which enhances its performance.\n   - **Sequence Packing**: Images are flattened along the sequence dimension and concatenated to process multiple images efficiently within a single batch. A block-diagonal mask ensures no attention leakage between patches from different images.\n   - **RoPE-2D**: Traditional position embeddings are replaced with relative, rotary position encodings (RoPE-2D) in the self-attention layers. This allows the model to handle variable image sizes more effectively without the need for interpolation, which can degrade performance.\n\n3. **Integration with Multimodal Decoder**: The vision encoder is linked to the multimodal decoder via a two-layer fully connected network. This network transforms the output of the vision encoder into the input embedding size required by the decoder. The image tokens are treated similarly to text tokens by the multimodal decoder, which uses RoPE-1D positional encodings for all tokens.\n\n4. **Performance**: The Pixtral vision encoder significantly outperforms other models in tasks requiring fine-grained document understanding while maintaining parity for natural images. It is particularly effective in settings that require detailed visual comprehension, such as chart and document understanding.\n\nThese architectural choices and modifications enable Pixtral 12B to flexibly process images at various resolutions and aspect ratios, making it highly versatile for complex multimodal applications."
        }
      ],
      "object": "entry",
      "created_at": "2025-05-27T13:16:11.239496Z",
      "completed_at": "2025-05-27T13:16:17.211241Z",
      "id": "msg_06835bb1b3d47ca580001b213d836798",
      "agent_id": "ag_06835bb196f9720680004fb1873efbae",
      "model": "mistral-medium-2505",
      "role": "assistant"
    }
  ],
  "usage": {
    "prompt_tokens": 196,
    "completion_tokens": 485,
    "total_tokens": 3846,
    "connector_tokens": 3165,
    "connectors": {
      "document_library": 1
    }
  },
  "object": "conversation.response"
}
py
image_agent = client.beta.agents.create(
    model="mistral-medium-2505",
    name="Image Generation Agent",
    description="Agent used to generate images.",
    instructions="Use the image generation tool when you have to create images.",
    tools=[{"type": "image_generation"}],
    completion_args={
        "temperature": 0.3,
        "top_p": 0.95,
    }
)
typescript
let imageAgent = await client.beta.agents.create({
    model:"mistral-medium-2505",
    name:"Image Generation Agent",
    description:"Agent used to generate images.",
    instructions:"Use the image generation tool when you have to create images.",
    tools:[{
        type: "image_generation"
    }],
    completionArgs:{
        temperature: 0.3,
        topP: 0.95,
    }
});
bash
curl --location "https://api.mistral.ai/v1/agents" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
     "model": "mistral-medium-2505",
     "name": "Image Generation Agent",
     "description": "Agent used to generate images.",
     "instructions": "Use the image generation tool when you have to create images.",
     "tools": [
       {
         "type": "image_generation"
       }
     ],
     "completion_args": {
       "temperature": 0.3,
       "top_p": 0.95
     }
  }'

json
{
  "model": "mistral-medium-2505",
  "name": "Image Generation Agent",
  "description": "Agent used to generate images.",
  "id": "ag_068359b1d997713480003c77113b8119",
  "version": 0,
  "created_at": "2025-05-27T10:59:41.602844Z",
  "updated_at": "2025-05-27T10:59:41.602846Z",
  "instructions": "Use the image generation tool when you have to create images.",
  "tools": [
    {
      "type": "image_generation"
    }
  ],
  "completion_args": {
    "stop": null,
    "presence_penalty": null,
    "frequency_penalty": null,
    "temperature": 0.3,
    "top_p": 0.95,
    "max_tokens": null,
    "random_seed": null,
    "prediction": null,
    "response_format": null,
    "tool_choice": "auto"
  },
  "handoffs": null,
  "object": "agent"
}

py
response = client.beta.conversations.start(
    agent_id=image_agent.id,
    inputs="Generate an orange cat in an office."
)
typescript
let conversation = await client.beta.conversations.start({
      agentId: imageAgent.id,
      inputs:"Generate an orange cat in an office.",
      //store:false
});
bash
curl --location "https://api.mistral.ai/v1/conversations" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
     "inputs": "Generate an orange cat in an office.",
     "stream": false,
     "agent_id": "<agent_id>"
  }'
json
{
  "conversation_id": "conv_068359b1dc6f74658000000a358b2357",
  "outputs": [
    {
      "name": "image_generation",
      "object": "entry",
      "type": "tool.execution",
      "created_at": "2025-05-27T10:59:53.092347Z",
      "completed_at": "2025-05-27T10:59:56.436333Z",
      "id": "tool_exec_068359b2917a7117800018b42bf8dc39"
    },
    {
      "content": [
        {
          "text": "Here is your image: an orange cat in an office.\n\n",
          "type": "text"
        },
        {
          "tool": "image_generation",
          "file_id": "933c5b5a-1c47-4cdd-84f6-f32526bd161b",
          "type": "tool_file",
          "file_name": "image_generated_0",
          "file_type": "png"
        }
      ],
      "object": "entry",
      "type": "message.output",
      "created_at": "2025-05-27T10:59:57.718377Z",
      "completed_at": "2025-05-27T10:59:58.818205Z",
      "id": "msg_068359b2db7e74eb8000d11444e03eb8",
      "agent_id": "ag_068359b1d997713480003c77113b8119",
      "model": "mistral-medium-2505",
      "role": "assistant"
    }
  ],
  "usage": {
    "prompt_tokens": 129,
    "total_tokens": 292,
    "completion_tokens": 94,
    "connector_tokens": 69,
    "connectors": {
      "image_generation": 1
    }
  },
  "object": "conversation.response"
}
py

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

**Delete a Library**
```

Example 3 (unknown):
```unknown
**Delete a Document**
```

Example 4 (unknown):
```unknown
</TabItem>
</Tabs>

<details>
    <summary><b>Contents</b></summary>
```

---

## Delete an access level.

**URL:** llms-txt#delete-an-access-level.

Source: https://docs.mistral.ai/api/#tag/libraries_share_delete_v1

del /v1/libraries/{library_id}/share

---

## Delete a document.

**URL:** llms-txt#delete-a-document.

Source: https://docs.mistral.ai/api/#tag/libraries_documents_delete_v1

del /v1/libraries/{library_id}/documents/{document_id}

---

## Delete a library and all of it's document.

**URL:** llms-txt#delete-a-library-and-all-of-it's-document.

Source: https://docs.mistral.ai/api/#tag/libraries_delete_v1

del /v1/libraries/{library_id}

---

## Delete File

**URL:** llms-txt#delete-file

Source: https://docs.mistral.ai/api/#tag/files_api_routes_delete_file

del /v1/files/{file_id}

---

## Delete Model

**URL:** llms-txt#delete-model

Source: https://docs.mistral.ai/api/#tag/delete_model_v1_models__model_id__delete

del /v1/models/{model_id}

---

## Detailed information about a specific Library.

**URL:** llms-txt#detailed-information-about-a-specific-library.

Source: https://docs.mistral.ai/api/#tag/libraries_get_v1

get /v1/libraries/{library_id}

---

## Document AI QnA

**URL:** llms-txt#document-ai-qna

The Document QnA capability combines OCR with large language model capabilities to enable natural language interaction with document content. This allows you to extract information and insights from documents by asking questions in natural language.

**The workflow consists of two main steps:**

<div style={{ textAlign: 'center' }}>
  <img
    src="/img/document_qna.png"
    alt="Document QnA Graph"
    width="800"
    style={{ borderRadius: '15px' }}
  />
</div>

1. Document Processing: OCR extracts text, structure, and formatting, creating a machine-readable version of the document.

2. Language Model Understanding: The extracted document content is analyzed by a large language model. You can ask questions or request information in natural language. The model understands context and relationships within the document and can provide relevant answers based on the document content.

**Key capabilities:**
- Question answering about specific document content
- Information extraction and summarization
- Document analysis and insights
- Multi-document queries and comparisons
- Context-aware responses that consider the full document

**Common use cases:**
- Analyzing research papers and technical documents
- Extracting information from business documents
- Processing legal documents and contracts
- Building document Q&A applications
- Automating document-based workflows

The examples below show how to interact with a PDF document using natural language:

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

from mistralai import Mistral

---

## Document Annotation response format

**URL:** llms-txt#document-annotation-response-format

class Document(BaseModel):
  language: str
  chapter_titles: list[str]
  urls: list[str]
python

from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, ResponseFormat
from mistralai.extra import response_format_from_pydantic_model

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

**Examples:**

Example 1 (unknown):
```unknown
**Start the completion**

Next, use the Mistral AI python client to make a request and ensure the response adheres to the defined structures using `bbox_annotation_format` and `document_annotation_format` set to the corresponding pydantic models:
```

---

## Download File

**URL:** llms-txt#download-file

Source: https://docs.mistral.ai/api/#tag/files_api_routes_download_file

get /v1/files/{file_id}/content

---

## download the validation and reformat script

**URL:** llms-txt#download-the-validation-and-reformat-script

wget https://raw.githubusercontent.com/mistralai/mistral-finetune/main/utils/reformat_data.py

---

## Download using the ToolFileChunk ID

**URL:** llms-txt#download-using-the-toolfilechunk-id

file_bytes = client.files.download(file_id=file_chunk.file_id).read()

---

## Email

**URL:** llms-txt#email

**Contents:**
- Evaluation
  - Include a confidence score

{insert customer email here}

{
  "Summaries": [
    {
      "Summary": "The author discusses their early experiences with programming and writing, starting with writing short stories and programming on an IBM 1401 in 9th grade. They then moved on to working with microcomputers, building their own from a Heathkit, and eventually convincing their father to buy a TRS-80 in 1980. They wrote simple games, a program to predict rocket flight trajectories, and a word processor.",
      "Confidence": 0.9
    },
    {
      "Summary": "The author began college as a philosophy major, but found it to be unfulfilling and switched to AI. They were inspired by a novel and a PBS documentary, as well as the potential for AI to create intelligent machines like those in the novel. Despite this excitement, they eventually realized that the traditional approach to AI was flawed and shifted their focus to Lisp.",
      "Confidence": 0.85
    },
    {
      "Summary": "The author briefly worked at Interleaf, where they found that their Lisp skills were highly valued. They eventually left Interleaf to return to RISD, but continued to work as a freelance Lisp hacker. While at RISD, they started painting still lives in their bedroom at night, which led to them applying to art schools and eventually attending the Accademia di Belli Arti in Florence.",
      "Confidence": 0.9
    }
  ]
}

You are a summarization system that can provide summaries with associated confidence scores.
In clear and concise language, provide three short summaries of the following essay, along with their confidence scores.
You will only respond with a JSON object with the key Summary and Confidence. Do not provide explanations.

**Examples:**

Example 1 (unknown):
```unknown
</details>


#### Strategies we used:
- **Providing facts**: Incorporating facts into prompts can be useful for developing customer support bots. It's important to use clear and concise language when presenting these facts. This can help the LLM to provide accurate and quick responses to customer queries.

## Evaluation
There are many ways to evaluate LLM outputs. Here are three approaches for your reference: include a confidence score, introduce an evaluation step, or employ another LLM for evaluation.

### Include a confidence score
We can include a confidence score along with the generated output.

<table class="prompt-example">
    <tr>
        <td>Assistant</td>
        <td>
```

Example 2 (unknown):
```unknown
</td>
    </tr>
</table>


<details>
<summary><b>Prompt</b></summary>
```

---

## Essay:

**URL:** llms-txt#essay:

{insert essay text here}

---

## evaluate code generation

**URL:** llms-txt#evaluate-code-generation

pass_at_1, results = code_eval.compute(references=refs, predictions=preds)

---

## Evaluation metric:

**URL:** llms-txt#evaluation-metric:

---

## Evaluation rubrics:

**URL:** llms-txt#evaluation-rubrics:

---

## example output:

**URL:** llms-txt#example-output:

---

## example output

**URL:** llms-txt#example-output

---

## example using code_eval:

**URL:** llms-txt#example-using-code_eval:

pass_at_1, results = code_eval.compute(
    references=['assert is_odd(5) == True'], 
    predictions=[['def is_odd(n):\n    return n & 1 != 0']], 
    k=[1])

---

## Facts

**URL:** llms-txt#facts

30-year fixed-rate: interest rate 6.403%, APR 6.484%
20-year fixed-rate: interest rate 6.329%, APR 6.429%
15-year fixed-rate: interest rate 5.705%, APR 5.848%
10-year fixed-rate: interest rate 5.500%, APR 5.720%
7-year ARM: interest rate 7.011%, APR 7.660%
5-year ARM: interest rate 6.880%, APR 7.754%
3-year ARM: interest rate 6.125%, APR 7.204%
30-year fixed-rate FHA: interest rate 5.527%, APR 6.316%
30-year fixed-rate VA: interest rate 5.684%, APR 6.062%

---

## file={

**URL:** llms-txt#file={

---

## "file_name": "uploaded_file.pdf",

**URL:** llms-txt#"file_name":-"uploaded_file.pdf",

---

## For a real problem, C should be properly cross validated and the confusion matrix analyzed

**URL:** llms-txt#for-a-real-problem,-c-should-be-properly-cross-validated-and-the-confusion-matrix-analyzed

clf = LogisticRegression(random_state=0, C=1.0, max_iter=500).fit(
    train_x, train_y.to_list()
)

---

## for each test case

**URL:** llms-txt#for-each-test-case

# define user message
    user_message = prompt_template.format(medical_notes=prompts[name]["medical_notes"])

# run LLM
    response = json.loads(run_mistral(user_message))

# calculate accuracy rate for this test case
    accuracy_rates.append(
        compare_json_objects(response, prompts[name]["golden_answer"])
    )

---

## Full Cookbook

**URL:** llms-txt#full-cookbook

**Contents:**
- Endpoints & Models
- FIM
  - Codestral
- Instruct Following
  - Codestral
  - Devstral
- Integrations
  - Codestral Integrations

You can find a comprehensive cookbook exploring Citations and References leveraging RAG with Wikipedia [here](https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/rag/mistral-reference-rag.ipynb).  
This template will help get started with web search and document grounding with citations.

[Coding]
Source: https://docs.mistral.ai/docs/capabilities/coding

LLMs are powerfull tools for text generation, and they also show great performance in code generation for multiple tasks, both for code completion, code generation and agentic tool use for semi-automated software development.

We provide 2 major families of llms for coding:
- **Codestral**: Specifically trained for Code Generation and FIM.
- **Devstral**: Specifically trained for Agentic Tool Use for Software Development.

Note that we also provide **Codestral Embed**, for semantic search and embedding code databases, repositories, and powering coding assistants with state-of-the-art retrieval. Learn more about it [here](https://docs.mistral.ai/capabilities/embeddings/code_embeddings).

## Endpoints & Models
We provide 2 main endpoints:
- `https://api.mistral.ai/v1/fim/completions`: [Fill-in-the-middle](#fim), for code completion and code generation; supporting `codestral-latest`.
- `https://api.mistral.ai/v1/chat/completions`: [Instruction following](#instruct-following), for coding and agentic tool use; supporting `codestral-latest`, `devstral-small-latest` and `devstral-medium-latest`.

With this feature, users can define the starting point of the code using a `prompt`, and the ending point of the code using an optional `suffix` and an optional `stop`. The FIM model will then generate the code that fits in between, making it ideal for tasks that require a specific piece of code to be generated.

:::tip[ ]
We also provide the `min_tokens` and `max_tokens` sampling parameters, which are particularly useful for code generation as it allows you to set the minimum and maximum number of tokens that should be produced. This is especially useful when FIM models decide to produce no tokens at all, or are overly verbose, allowing developers to enforce completions within a specific range if they are needed.
:::

### Codestral
Codestral is a cutting-edge generative model that has been specifically designed and optimized for code generation tasks, including fill-in-the-middle and code completion. Codestral was trained on 80+ programming languages, enabling it to perform well on both common and less common languages.

:::important[ ]
We currently offer two domains for Codestral endpoints, both providing FIM and instruct routes:

| Domain  | Features |
| ------------- | ------------- |
| codestral.mistral.ai | - Monthly subscription based, currently free to use <br/> - Requires a new key for which a phone number is needed |
| api.mistral.ai  | - Allows you to use your existing API key and you can pay to use Codestral <br/> - Ideal for business use |

Wondering which endpoint to use?
- If you're a user, wanting to query Codestral as part of an IDE plugin, codestral.mistral.ai is recommended.
- If you're building a plugin, or anything that exposes these endpoints directly to the user, and expect them to bring their own API keys, you should also target codestral.mistral.ai
- For all other use cases, api.mistral.ai will be better suited

*This guide uses api.mistral.ai for demonstration.*
:::

Below we present three examples:

#### Example 1: Fill in the middle
Originally, these models are designed to complete code in-between 2 points: a prefix (here called `prompt`) and a `suffix`, generating the code in-between.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="curl" label="curl" default>

#### Example 2: Completion
However, you can also use the model for pure code completion, by only providing a `prompt` and no `suffix`.
<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="curl" label="curl">

#### Example 3: Stop tokens
You can also use stop tokens to control the generation of the model when it generates specific strings.
:::tip[ ]
We recommend adding stop tokens for IDE autocomplete integrations to prevent the model from being too verbose.
:::

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="curl" label="curl">

## Instruct Following

We also provide the instruct chat endpoint of Codestral with the same model `codestral-latest`.  
The only difference is the endpoint used; so you can leverage powerfull code completion with instruct and chat use cases.

However we also provide `devstral-small-latest` and `devstral-medium-latest` for agentic tool use for software development, this family of models is specifically trained to navigate code bases and leverage tool usage for diverse tasks.

Here is an example of how to use the instruct endpoint of Codestral, it's perfect for specific **code generation** of specific snippets or **code completion** while **following instructions**; so you can better guide generation and exchange with a powerfull coding model.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="curl" label="curl">

While Codestral is designed for code generation and FIM, Devstral is a cutting-edge generative model that has been specifically designed and optimized for **agentic tool use for software development**, it can leverage function calling to navigate code bases and call the right tools to perform specific tasks for semi-automated software development.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

<TabItem value="curl" label="curl">

### Codestral Integrations

<details>
<summary><b>Integration with continue.dev</b></summary>

Continue.dev supports both Codestral base for code generation and Codestral Instruct for chat.

<iframe width="560" height="315" width="100%" src="https://www.youtube.com/embed/mjltGOJMJZA?si=Tmf0kpPn3hVJ0CaM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

#### How to set up Codestral with Continue

**Here is a step-by-step guide on how to set up Codestral with Continue using the Mistral AI API:**

1. Install the Continue VS Code or JetBrains extension following the instructions [here](https://docs.continue.dev/quickstart). 
Please make sure you install Continue version >v0.8.33.

- Click on the Continue extension iron on the left menu. Select `Mistral API` as a provider, select `Codestral` as a model. 
- Click "Get API Key" to get Codestral API key. 
- Click "Add model", which will automatically populate the config.json.

<img src="/img/guides/codestral1.png" alt="drawing" width="300"/>

2. (alternative) Manually edit config.json 
- Click on the gear icon in the bottom right corner of the Continue window to open `~/.continue/config.json` (MacOS) /  `%userprofile%\.continue\config.json` (Windows)
- Log in and request a Codestral API key on Mistral AI's La Plateforme [here](https://console.mistral.ai/codestral)
- To use Codestral as your model for both `autocomplete` and `chat`, replace  `[API_KEY]` with your Mistral API key below and add it to your `config.json` file:

If you run into any issues or have any questions, please join our Discord and post in `#help` channel [here](https://discord.gg/EfJEfdFnDQ)
</details>

<details>
<summary><b>Integration with Tabnine</b></summary>

Tabnine supports Codestral Instruct for chat.

<iframe width="560" height="315" width="100%" src="https://www.youtube.com/embed/pFa4NLK9Lbw?si=7tsfFUsOyllkwl-M" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

#### How to set up Codestral with Tabnine

##### What is Tabnine Chat? 
Tabnine Chat is a code-centric chat application that runs in the IDE and allows developers
 to interact with Tabnine’s AI models in a flexible, free-form way, using natural language. 
Tabnine Chat also supports dedicated quick actions that use predefined prompts optimized
 for specific use cases.

##### Getting started
To start using Tabnine Chat, first [launch](https://docs.tabnine.com/main/getting-started/getting-the-most-from-tabnine-chat/launch) it in your IDE (VSCode, JetBrains, or Eclipse). 
Then, learn how to [interact](https://docs.tabnine.com/main/getting-started/getting-the-most-from-tabnine-chat/interact) with Tabnine Chat, for example, how to ask questions or give 
instructions. Once you receive your response, you can [read, review, and apply](https://docs.tabnine.com/main/getting-started/getting-the-most-from-tabnine-chat/consume) it within 
your code.

##### Selecting Codestral as Tabnine Chat App model

In the Tabnine Chat App, use the [model selector](https://docs.tabnine.com/main/getting-started/getting-the-most-from-tabnine-chat/switching-between-chat-ai-models) to choose *Codestral*.

<details>
<summary><b>Integration with LangChain</b></summary>

LangChain provides support for Codestral Instruct. Here is how you can use it in LangChain:

**Examples:**

Example 1 (python):
```python
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

model = "codestral-latest"
prompt = "def fibonacci(n: int):"
suffix = "n = int(input('Enter a number: '))\nprint(fibonacci(n))"

response = client.fim.complete(
    model=model,
    prompt=prompt,
    suffix=suffix,
    temperature=0,
    # min_tokens=1, # Uncomment to enforce completions to at least 1 token
)

print(
    f"""
{prompt}
{response.choices[0].message.content}
{suffix}
"""
)
```

Example 2 (bash):
```bash
curl --location 'https://api.mistral.ai/v1/fim/completions' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header "Authorization: Bearer $MISTRAL_API_KEY" \
--data '{
    "model": "codestral-latest",
    "prompt": "def f(",
    "suffix": "return a + b",
    "max_tokens": 64,
    "temperature": 0
}'
```

Example 3 (python):
```python
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

model = "codestral-latest"
prompt = "def is_odd(n): \n return n % 2 == 1 \ndef test_is_odd():"

response = client.fim.complete(model=model, prompt=prompt, temperature=0)

print(
    f"""
{prompt}
{response.choices[0].message.content}
"""
)
```

Example 4 (bash):
```bash
curl --location 'https://api.mistral.ai/v1/fim/completions' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header "Authorization: Bearer $MISTRAL_API_KEY" \
--data '{
    "model": "codestral-latest",
    "prompt": "def is_odd(n): \n return n % 2 == 1 \n def test_is_odd():", 
    "suffix": "",
    "max_tokens": 64,
    "temperature": 0
}'
```

---

## Getting the base64 string

**URL:** llms-txt#getting-the-base64-string

base64_image = encode_image(image_path)

---

## Get Batch Jobs

**URL:** llms-txt#get-batch-jobs

Source: https://docs.mistral.ai/api/#tag/jobs_api_routes_batch_get_batch_jobs

---

## Get Batch Job

**URL:** llms-txt#get-batch-job

Source: https://docs.mistral.ai/api/#tag/jobs_api_routes_batch_get_batch_job

get /v1/batch/jobs/{job_id}

---

## get data from hugging face

**URL:** llms-txt#get-data-from-hugging-face

ds = load_dataset("HuggingFaceH4/ultrachat_200k",split="train_gen")

---

## Get document info once processed

**URL:** llms-txt#get-document-info-once-processed

deleted_library = client.beta.libraries.delete(library_id=new_library.id)

---

## Get Signed Url

**URL:** llms-txt#get-signed-url

Source: https://docs.mistral.ai/api/#tag/files_api_routes_get_signed_url

get /v1/files/{file_id}/url

---

## How to contribute

**URL:** llms-txt#how-to-contribute

**Contents:**
- Join our Ambassador Program
- Contributing to the official documentation
- Contributing to the code clients
  - Python client
  - JavaScript client
- Contributing to the community
  - Mistral Cookbook
  - Discord
  - Community outreach
- What is your task?

Thank you for your interest in contributing to Mistral AI. We welcome everyone who wishes to contribute and we appreciate your time and effort!

## Join our Ambassador Program

:::tip[ ]
Head to [Ambassador Program](../ambassador/) to learn more and apply! 
:::

## Contributing to the official documentation

If you are interested in contributing to our [official docs](https://docs.mistral.ai/), please submit a PR at [https://github.com/mistralai/platform-docs-public](https://github.com/mistralai/platform-docs-public).

You can easily help by:
- fix a typo
- clarify a section
- document an underdocumented feature
- update a section that should have been updated
- ...

## Contributing to the code clients

### Python client
Submit a PR to improve our Python client: [https://github.com/mistralai/client-python](https://github.com/mistralai/client-python).

The easiest way to get started is to take a look at the [issue list](https://github.com/mistralai/client-python/issues) and see which one you might be interested in helping.

### JavaScript client
Submit a PR to improve our JavaScript client: [https://github.com/mistralai/client-js](https://github.com/mistralai/client-js)

The easiest way to get started is to take a look at the [issue list](https://github.com/mistralai/client-js/issues) and see which one you might be interested in helping.

## Contributing to the community

### Mistral Cookbook
If you have cool examples showcasing Mistral models, whether you are using Mistral API or open-weight models, feel free to share them by submitting a PR to our [cookbook repo](https://github.com/mistralai/cookbook).

We are grateful to anyone who shares resources and assists us in addressing questions on our [Discord](https://discord.gg/mistralai) community.

### Community outreach
A valuable way to support Mistral AI is by engaging in active communication in the community. This can include activities like sharing on social media, writing blog articles, creating tutorial videos demonstrating your work using Mistral AI API or models. Feel free to tag us on social media, and we'll be there to support you!

[Evaluation]
Source: https://docs.mistral.ai/docs/guides/evaluation

<a target="_blank" href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/evaluation/evaluation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Many companies face the challenge of evaluating whether a Large Language Model (LLM) is suitable for their specific use cases and determining which LLMs offer the best performance. It is essential to assess whether one model can effectively replace another and ensure that the chosen LLM meets the companies’ unique requirements. However, the process of evaluating LLMs can be complex and confusing. This guide provides a comprehensive approach to evaluating LLMs for various applications. We will start by discussing the goal of evaluation, the creation of evaluation datasets, and then we will dive into three primary evaluation methods with detailed Python code walkthroughs in notebooks.

- What exact task do you want to solve?
- How to create an evaluation dataset?
- How to evaluate?  
    - Metrics-based evaluation
    - LLM-based evaluation
    - Human-based evaluation

## What is your task? 
Before we get started with the evaluation process, it is important to think about what exact task you want to solve with the LLM. It is crucial that the task is precisely defined.

Some bad examples:
- Be a helpful assistant (What does “helpful” mean in your context?)
- Convert the text into code (What type of text and code?)
- Improve the quality of the documents (What is high vs. low quality?)

Some good examples:
- Be an assistant for customer complaints. Make sure to answer complaints in a polite and helpful way and give concise answers that don’t exceed 3 sentences. 
- Make sure to notify the user if you do not know the answer.
- Convert pseudo code into Python code 
- Rephrase the documents by improving all spelling and grammatical errors and give the text a more professional tone
- Extract all relevant information from medical records (which use case would be subject to specific applicable regulations)
- Summarize various types of documents, such as legal agreements, news articles, or scientific papers

Different goals and use cases may require different evaluation strategies. Some use cases may prioritize accuracy, while others may emphasize brevity and helpfulness. Therefore, it is crucial to know exactly what task you want to solve before starting an evaluation process.

## How to create an evaluation dataset?
There are numerous public benchmarks available for evaluating Large Language Models (LLMs) such as MMLU, GSMK8, and others. The first question to consider is whether these existing benchmarks can be used for your specific use cases. If applicable, then start with the existing benchmark dataset.

However, we often observe that existing benchmarks are academic in nature and may not cover real-world customer use cases or only address a limited scope. Therefore, it is often preferable to create your own evaluation dataset.

Once you have established your goals and determined whether existing benchmarks are suitable, you can proceed to create custom evaluation datasets tailored to your specific use cases.

In many cases, it is important to create labeled data. For example, if your task is to use an LLM to extract information from medical records subject to specific applicable regulations, you can use human annotation to label the data and get the golden answer including all the information you would like to extract, e.g.,  `{"age": 60, "gender": "male", "diagnosis": "diabetes", "weight": 210, "smoking": "yes"}`.

How much data annotation do you need? It depends on the task. You should always prioritize quality over quantity. If you have high quality data, 100 test cases can be enough to provide a meaningful signal.

Additionally, It is recommended to have a discriminative benchmark, meaning that it’s neither too easy, nor too hard. If every model being tested achieves over 90% accuracy, the differences between them might not be meaningful, indicating the need for a better benchmark.  The goal is to create data that can differentiate models effectively. Creating discriminative data can be challenging, so focus on your specific task. Potentially multiple models perform well enough for your needs. A good evaluation dataset should lead to performances that are in agreement with model sizes, ie, larger models perform on par or better than smaller counterparts (unless the smaller models have been specifically trained for your purposes).

Here are some examples of various prompts and golden answers across different capabilities:

| Task               | Prompt/Context| Golden Answer/Criteria|
|--------------------|:--------------------|:--------------------|
| Information extraction                | A 60-year-old male patient, Mr. Johnson, presented with symptoms of increased thirst, frequent urination, fatigue, and unexplained weight loss. Upon evaluation, he was diagnosed with diabetes, confirmed by elevated blood sugar levels. Mr. Johnson's weight is 210 lbs. He has been prescribed Metformin to be taken twice daily with meals. It was noted during the consultation that the patient is a current smoker.| `{"age": 60, "gender": "male", "diagnosis": "diabetes", "weight": 210, "smoking": "yes"}` |
|Python code generation|Write a python function to check whether the given number is odd or not using bitwise operator.|`assert is_odd(5) == True` |
|Summarization| Summarize the following into a short paragraph for 3rd graders: "Water is a chemical compound composed of two hydrogen atoms bonded to one oxygen atom, with the chemical formula H2O. The hydrogen atoms are bonded to the oxygen atom via polar covalent bonds, which means that the electrons in the bond are not shared equally between the atoms, resulting in a slight negative charge on the oxygen atom and a slight positive charge on the hydrogen atoms. This gives water molecules a bent, or V-shaped, structure, and allows them to form hydrogen bonds with other water molecules, which are responsible for many of water's unique properties, such as its high boiling point and surface tension." |Water is a special substance that is made up of two tiny parts called atoms, specifically, two hydrogen atoms and one oxygen atom. These atoms are stuck together in a way that creates a bent shape, which makes water have some amazing properties. For example, water can stick to itself, which helps it stay together as a liquid and not just turn into a gas. This also makes water have a high boiling point, which means it needs a lot of heat to turn into a gas. Water's unique properties make it an important part of our lives and our planet.|
|Multilingual|Translate this college homework assignment into advanced Spanish for an advanced writing course: "Write a personal essay of 500 words about a crucial moment in your family, describing the event, its impact on family relationships, and the emotions and thoughts that surrounded it. Your essay should have a clear structure, with an introduction, body, and conclusion, and use expressive and precise language. It will be evaluated based on clarity, coherence, depth, and originality."|Escriba un ensayo personal de 500 palabras sobre un momento crucial en su familia, describiendo el evento, su impacto en las relaciones familiares, y las emociones y pensamientos que lo rodearon. Su ensayo debe tener una estructura clara, con una introducción, un cuerpo y una conclusión, y utilizar un lenguaje expresivo y preciso. Será evaluado sobre la base de claridad, coherencia, profundidad y originalidad.|
|Chat|You are an 8th grade tutor for a Latin class, your job is to help students memorize their weekly word list test. You don’t give the answers to your students but instead help them get to the answer themselves. Begin your tutoring session by asking your student what word list do they need help memorizing. Remember to be encouraging and ask students to "show their work" i.e. articulate how they came up with that answer.|Hello there! I hope you're having a great day. As your Latin tutor, I'm here to help you with your weekly word list test. So, let's get started! Could you please tell me which word list you need help memorizing this week? Remember, the goal is not just to remember the words, but also to understand their meanings and how to use them in context. <br/> Once you've chosen a word, I'd like you to try to recall its meaning or use it in a sentence. Then, I'd love for you to "show your work" by explaining how you came up with that answer. This will help reinforce your understanding and make it easier to remember. Does that sound good? Let's dive in!

In some cases, you may not be able to create labeled data easily. You may use an LLM to generate the necessary data. For example, you may ask an LLM to generate questions and answers based on a given text. Here are two prompt templates for generating questions and answers:

- Prompt template for generating questions based on the context:

- Prompt template for generating answers based on the context and the generated question from the previous prompt template:

However, LLM-generated data usually requires further refinement and filtering. It is preferable to have questions derived from real-world sources, especially from experts. Nevertheless, LLM-generated data can still be helpful when real data is difficult to obtain or scarce.

## Metrics-based evaluation
Metrics-based evaluation is the standard approach in many public benchmark evaluations. There are a number of metrics that can be used to evaluate and compare the model-generated answers with the golden answers. The most popular ones are exact match accuracy rate, recall, precision, F1 score.
- **Exact Match Accuracy Rate**: This metric measures the proportion of model-generated answers that perfectly match the golden answers. It offers a strict evaluation of the model's ability to generate precise responses.
- **Recall**: Recall calculates the fraction of relevant information from the golden answers that is successfully retrieved by the model-generated answers. A higher recall indicates that the model can capture more relevant information.
- **Precision**: Precision assesses the fraction of relevant information in the model-generated answers. A higher precision means that the model generates more accurate and relevant responses.
- **F1 Score**: The F1 score is the harmonic mean of recall and precision, providing a balanced evaluation of the model's performance in terms of both metrics.

Apart from these popular metrics, there are other NLP evaluation metrics such as BLEU (Bilingual Evaluation Understudy) and ROUGE (Recall-Oriented Understudy for Gisting Evaluation). BLEU is primarily used for machine translation tasks, while ROUGE is commonly applied for text summarization. However, due to concerns regarding their reliability and the potential for misleading results, we do not recommend.

<details>
<summary><b> Example 1: evaluate information extraction with accuracy rate</b></summary>

### Example 1: evaluate information extraction with accuracy rate

This first example involves extracting patient information from medical notes. To perform the evaluation, both the medical notes and the correct/expected answer (referred to as the "golden answer") are required in the evaluation data. Here are two example test cases:

#### How to evaluate?

- Step 1: Define prompt template

We have designed a prompt that incorporates the medical notes as context. Additionally, we expect the model to provide output in a JSON format following a predefined JSON schema, ensuring that the model produces the desired output accurately. It is worth noting that when calling our models, we specified the response format as `{"type": "json_object"}` to ensure consistent JSON output.

from mistralai import Mistral

def run_mistral(user_message, model="mistral-large-latest"):
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    messages = [{"role": "user", "content": user_message}]
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return chat_response.choices[0].message.content

**Examples:**

Example 1 (unknown):
```unknown
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge. Generate {num_questions_per_chunk} questions based on the context. The questions should be diverse in nature across the document. Restrict the questions to the context information provided.
```

Example 2 (unknown):
```unknown
Context information is below
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge,
answer the query.
Query: {generated_query_str}
Answer:
```

Example 3 (py):
```py
prompts = {
    "Johnson": {
        "medical_notes": "A 60-year-old male patient, Mr. Johnson, presented with symptoms of increased thirst, frequent urination, fatigue, and unexplained weight loss. Upon evaluation, he was diagnosed with diabetes, confirmed by elevated blood sugar levels. Mr. Johnson's weight is 210 lbs. He has been prescribed Metformin to be taken twice daily with meals. It was noted during the consultation that the patient is a current smoker. ",
        "golden_answer": {
            "age": 60,
            "gender": "male",
            "diagnosis": "diabetes",
            "weight": 210,
            "smoking": "yes",
        },
    },
    "Smith": {
        "medical_notes": "Mr. Smith, a 55-year-old male patient, presented with severe joint pain and stiffness in his knees and hands, along with swelling and limited range of motion. After a thorough examination and diagnostic tests, he was diagnosed with arthritis. It is important for Mr. Smith to maintain a healthy weight (currently at 150 lbs) and quit smoking, as these factors can exacerbate symptoms of arthritis and contribute to joint damage.",
        "golden_answer": {
            "age": 55,
            "gender": "male",
            "diagnosis": "arthritis",
            "weight": 150,
            "smoking": "yes",
        },
    },
}
```

---

## If local document, upload and retrieve the signed url

**URL:** llms-txt#if-local-document,-upload-and-retrieve-the-signed-url

---

## Initialize the Mistral client

**URL:** llms-txt#initialize-the-mistral-client

client = Mistral(api_key=api_key)

---

## Initialize the model

**URL:** llms-txt#initialize-the-model

**Contents:**
- Run on the cloud
- Set-up
- SkyPilot Configuration
  - Test it out!
- Usage Quotas
- Deploying
- Using the API
  - With chat-compatible endpoint

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3", dtype="bfloat16", max_model_len=20000, gpu_memory_utilization=0.9)

def run(prompt: str, temperature: float = 0.8, top_p: float = 0.75, top_k: int = 40, max_tokens: int = 256, frequency_penalty: int = 1):
  
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty
    )

outputs = llm.generate([item.prompt], sampling_params)

generated_text = []
    for output in outputs:
        generated_text.append(output.outputs[0].text)

return {"result": generated_text}
bash
cerebrium deploy
CURL
curl --location --request POST 'https://api.cortex.cerebrium.ai/v4/p-<YOUR PROJECT ID>/mistral-vllm/run' \
--header 'Authorization: Bearer <YOUR TOKEN HERE>' \
--header 'Content-Type: application/json' \
--data-raw '{
    "prompt: "What is the capital city of France?"
}'
json
{
  "run_id": "nZL6mD8q66u4lHTXcqmPCc6pxxFwn95IfqQvEix0gHaOH4gkHUdz1w==",
  "message": "Finished inference request with run_id: `nZL6mD8q66u4lHTXcqmPCc6pxxFwn95IfqQvEix0gHaOH4gkHUdz1w==`",
  "result": {
    "result": ["\nA: Paris"]
  },
  "status_code": 200,
  "run_time_ms": 151.24988555908203
}
bash
  curl https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/mistral/mistral-7b-instruct-v0.1 \
    -X POST \
    -H "Authorization: Bearer {API_TOKEN}" \
    -d '{ "messages": [{ "role": "user", "content": "[INST] 2 + 2 ? [/INST]" }]}'
  typescript
  async function run(model, prompt) {
    const messages = [
      { role: "user", content: prompt },
    ];

const response = await fetch(
      `https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/${model}`,
      {
        headers: { Authorization: "Bearer {API_TOKEN}" },
        method: "POST",
        body: JSON.stringify({ messages }),
      }
    );
    const result = await response.json();
    return result;
  }

run("@cf/mistral/mistral-7b-instruct-v0.1", "[INST] 2 + 2 ? [/INST]").then(
    (response) => {
      console.log(JSON.stringify(response));
    }
  );
  python
  import requests

API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/"
  headers = {"Authorization": "Bearer {API_TOKEN}"}

def run(model, prompt):
    input = {
      "messages": [
        { "role": "user", "content": prompt }
      ]
    }
    response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input)
    return response.json()

output = run("@cf/mistral/mistral-7b-instruct-v0.1", "[INST] 2 + 2 = ? [/INST]")
  print(output)
  python
{'result': {'response': '2 + 2 = 4.'}, 'success': True, 'errors': [], 'messages': []}
yaml
resources: 
  cloud: ${CLOUD_PROVIDER}
  accelerators: A10G:1
  ports: 
    - 8000

run: |
  docker run --gpus all -p 8000:8000 ghcr.io/mistralai/mistral-src/vllm:latest \
                   --host 0.0.0.0 \
                   --model mistralai/Mistral-7B-Instruct-v0.2 \
                   --tensor-parallel-size 1
yaml
resources: 
  cloud: ${CLOUD_PROVIDER}
  accelerators: A100-80GB:2
  ports: 
    - 8000

run: |
  docker run --gpus all -p 8000:8000 ghcr.io/mistralai/mistral-src/vllm:latest \
                   --host 0.0.0.0 \
                   --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
                   --tensor-parallel-size 2
yaml
resources: 
  cloud: ${CLOUD_PROVIDER}
  accelerators: A100-80GB:4
  ports: 
    - 8000

run: |
  docker run --gpus all -p 8000:8000 ghcr.io/mistralai/mistral-src/vllm:latest \
                   --host 0.0.0.0 \
                   --model mistralai/Mixtral-8x22B-Instruct-v0.1 \
                   --tensor-parallel-size 4
bash
sky launch -c mistral-7b mistral-7b-v0.1.yaml --region us-east-1
bash
sky status --ip mistral-7b

IP=$(sky status --ip cluster-name)

curl http://$IP:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "mistralai/Mistral-7B-v0.1",
      "prompt": "My favourite condiment is",
      "max_tokens": 25
  }'
bash
model=mistralai/Mistral-7B-Instruct-v0.3
bash
model=mistralai/Mixtral-8x22B-Instruct-v0.1
bash
model=mistralai/Mixtral-8x22B-Instruct-v0.1
bash
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data  \
    -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
    ghcr.io/huggingface/text-generation-inference:2.0.3 \
    --model-id $model
python
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

**Examples:**

Example 1 (unknown):
```unknown
## Run on the cloud
```

Example 2 (unknown):
```unknown
You will see your application deploy, install pip packages and download the model. Once completed it will output a CURL request you can use to call your endpoint. Just remember to end
the url with the function you would like to call - in this case /run.
```

Example 3 (unknown):
```unknown
You should then get a message looking like this:
```

Example 4 (unknown):
```unknown
[Deploy with Cloudflare Workers AI]
Source: https://docs.mistral.ai/docs/deployment/self-deployment/cloudflare

[Cloudflare](https://www.cloudflare.com/en-gb/) is a web performance and security company that provides content delivery network (CDN), DDoS protection, Internet security, and distributed domain name server services. Cloudflare launched Workers AI, which allows developers to run LLMs models powered by serverless GPUs on Cloudflare’s global network.

To learn more about Mistral models on Workers AI you can read the dedicated [Cloudflare documentation page](https://developers.cloudflare.com/workers-ai/models/mistral-7b-instruct-v0.1/).

## Set-up

To set-up Workers AI on Cloudflare, you need to create an account on the [Cloudflare dashboard](https://dash.cloudflare.com/), get your account ID, and generate a token with Workers AI permissions. You can then send a completion request:

<Tabs>
  <TabItem value="cloudflare-curl" label="curl" default>
```

---

## Instructions:

**URL:** llms-txt#instructions:

**Contents:**
- Summarize:
- Interesting Questions:
- Write a report
- Personalization

## Summarize:
In clear and concise language, summarize the key points and themes presented in the essay.

## Interesting Questions:
Generate three distinct and thought-provoking questions that can be asked about the content of the essay. For each question:
- After "Q: ", describe the problem
- After "A: ", provide a detailed explanation of the problem addressed in the question.
- Enclose the ultimate answer in <>.

## Write a report
Using the essay summary and the answers to the interesting questions, create a comprehensive report in Markdown format.

You are a mortgage lender customer service bot, and your task is to create personalized email responses to address customer questions. Answer the customer's inquiry using the provided facts below. Ensure that your response is clear, concise, and directly addresses the customer's question. Address the customer in a friendly and professional manner. Sign the email with "Lender Customer Support."

**Examples:**

Example 1 (unknown):
```unknown
</details>

#### Strategies we used:

- **Step-by-step instructions**: This strategy is inspired by the chain-of-thought prompting that enables LLMs to use a series of intermediate reasoning steps to tackle complex tasks. It's often easier to solve complex problems when we decompose them into simpler and small steps and it's easier for us to debug and inspect the model behavior. In our example, we break down the task into three steps: summarize, generate interesting questions, and write a report. This helps the language to think in each step and generate a more comprehensive final report.
- **Example generation**: We can ask LLMs to automatically guide the reasoning and understanding process by generating examples with the explanations and steps. In this example, we ask the LLM to generate three questions and provide detailed explanations for each question.
- **Output formatting**: We can ask LLMs to output in a certain format by directly asking "write a report in the Markdown format".

## Personalization
LLMs excel at personalization tasks as they can deliver content that aligns closely with individual users. In this example, we create personalized email responses to address customer questions.


<table class="prompt-example">
    <tr>
        <td>User</td>
        <td>Dear mortgage lender, <br/><br/>
        What's your 30-year fixed-rate APR, how is it compared to the 15-year fixed rate? <br/><br/>
        Regards, <br/>
        Anna</td>
    </tr>
    <tr>
        <td>Assistant</td>
        <td>
        Dear Anna, <br/><br/>
        Thank you for your inquiry regarding our fixed-rate mortgage options.
        Our current 30-year fixed-rate interest rate is 6.403% with an Annual Percentage Rate (APR) of 6.484%. In comparison, our 15-year fixed-rate interest rate is 5.705% with an APR of 5.848%.<br/><br/>
        As you can see, the 15-year fixed-rate option has a lower interest rate and APR compared to the 30-year fixed-rate option. This means that you would pay less in interest over the life of the loan if you choose the 15-year option. However, the monthly payments would be higher due to the shorter loan term.<br/><br/>
        Please let us know if you have any other questions or if we can assist you further in choosing the best mortgage option for your needs.<br/><br/>
        Best regards,<br/>
        Lender Customer Support
        </td>
    </tr>
</table>


<details>
<summary><b>Prompt</b></summary>
```

---

## integrations=[

**URL:** llms-txt#integrations=[

---

## List all libraries you have access to.

**URL:** llms-txt#list-all-libraries-you-have-access-to.

Source: https://docs.mistral.ai/api/#tag/libraries_list_v1

---

## List all of the access to this library.

**URL:** llms-txt#list-all-of-the-access-to-this-library.

Source: https://docs.mistral.ai/api/#tag/libraries_share_list_v1

get /v1/libraries/{library_id}/share

---

## List document in a given library.

**URL:** llms-txt#list-document-in-a-given-library.

Source: https://docs.mistral.ai/api/#tag/libraries_documents_list_v1

get /v1/libraries/{library_id}/documents

---

## List Files

**URL:** llms-txt#list-files

Source: https://docs.mistral.ai/api/#tag/files_api_routes_list_files

---

## List jobs

**URL:** llms-txt#list-jobs

curl https://api.mistral.ai/v1/fine_tuning/jobs \
--header "Authorization: Bearer $MISTRAL_API_KEY"

---

## make sure to install `langchain` and `langchain-mistralai` in your Python environment

**URL:** llms-txt#make-sure-to-install-`langchain`-and-`langchain-mistralai`-in-your-python-environment

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

api_key = os.environ["MISTRAL_API_KEY"]
mistral_model = "codestral-latest"
llm = ChatMistralAI(model=mistral_model, temperature=0, api_key=api_key)
llm.invoke([("user", "Write a function for fibonacci")])
py

**Examples:**

Example 1 (unknown):
```unknown
For a more complex use case of self-corrective code generation using the instruct Codestral tool use, check out this [notebook](https://github.com/mistralai/cookbook/blob/main/third_party/langchain/langgraph_code_assistant_mistral.ipynb) and this video:

<iframe width="560" height="315" width="100%" src="https://www.youtube.com/embed/zXFxmI9f06M?si=8ZEoqNVECVJQFcVA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

</details>

<details>
<summary><b>Integration with LlamaIndex</b></summary>

LlamaIndex provides support for Codestral Instruct and Fill In Middle (FIM) endpoints. Here is how you can use it in LlamaIndex:
```

---

## Make sure to install the huggingface_hub package before

**URL:** llms-txt#make-sure-to-install-the-huggingface_hub-package-before

**Contents:**
- Building the engine
- Deploying the engine
- Pre-requisites
- Getting started
  - Installing vLLM
  - Offline mode inference
  - Server mode inference
- Deploying with Docker
- Python
- Typescript

from huggingface_hub import InferenceClient

client = InferenceClient(model="http://127.0.0.1:8080")
client.text_generation(prompt="What is Deep Learning?")
typescript
async function query() {
    const response = await fetch(
        'http://127.0.0.1:8080/generate',
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json'},
            body: JSON.stringify({
                'inputs': 'What is Deep Learning?'
            })
        }
    );
}

query().then((response) => {
    console.log(JSON.stringify(response));
});

curl 127.0.0.1:8080/generate \
-X POST \
-d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
-H 'Content-Type: application/json'
bash
  huggingface-cli login --token $HF_TOKEN
  python
        from vllm import LLM
        from vllm.sampling_params import SamplingParams

model_name = "mistralai/Mistral-NeMo-Instruct-2407"
        sampling_params = SamplingParams(max_tokens=8192)

llm = LLM(
            model=model_name,
            tokenizer_mode="mistral",
            load_format="mistral",
            config_format="mistral",
        )

messages = [
            {
                "role": "user",
                "content": "Who is the best French painter. Answer with detailed explanations.",
            }
        ]

res = llm.chat(messages=messages, sampling_params=sampling_params)
        print(res[0].outputs[0].text)

python
        from vllm import LLM
        from vllm.sampling_params import SamplingParams

model_name = "mistralai/Mistral-Small-Instruct-2409"
        sampling_params = SamplingParams(max_tokens=8192)

llm = LLM(
            model=model_name,
            tokenizer_mode="mistral",
            load_format="mistral",
            config_format="mistral",
        )

messages = [
            {
                "role": "user",
                "content": "Who is the best French painter. Answer with detailed explanations.",
            }
        ]

res = llm.chat(messages=messages, sampling_params=sampling_params)
        print(res[0].outputs[0].text)

python
        from vllm import LLM
        from vllm.sampling_params import SamplingParams

model_name = "mistralai/Pixtral-12B-2409"
        max_img_per_msg = 3

sampling_params = SamplingParams(max_tokens=8192)
        llm = LLM(
            model=model_name,
            tokenizer_mode="mistral",
            load_format="mistral",
            config_format="mistral",
            limit_mm_per_prompt={"image": max_img_per_msg},
        )

urls = [f"https://picsum.photos/id/{id}/512/512" for id in ["1", "11", "111"]]

messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    ] + [{"type": "image_url", "image_url": {"url": f"{u}"}} for u in urls],
            },
        ]

res = llm.chat(messages=messages, sampling_params=sampling_params)
        print(res[0].outputs[0].text)
        bash
          vllm serve mistralai/Mistral-Nemo-Instruct-2407 \
            --tokenizer_mode mistral \
            --config_format mistral \
            --load_format mistral
          bash
                curl --location 'http://localhost:8000/v1/chat/completions' \
                    --header 'Content-Type: application/json' \
                    --header 'Authorization: Bearer token' \
                    --data '{
                        "model": "mistralai/Mistral-Nemo-Instruct-2407",
                        "messages": [
                          {
                            "role": "user",
                            "content": "Who is the best French painter? Answer in one short sentence."
                          }
                        ]
                      }'
                python
                import httpx

url = 'http://localhost:8000/v1/chat/completions'
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer token'
                }
                data = {
                    "model": "mistralai/Mistral-Nemo-Instruct-2407",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Who is the best French painter? Answer in one short sentence."
                        }
                    ]
                }

response = httpx.post(url, headers=headers, json=data)

print(response.json())

bash
          vllm serve mistralai/Mistral-Small-Instruct-2409 \
            --tokenizer_mode mistral \
            --config_format mistral \
            --load_format mistral
          bash
                curl --location 'http://localhost:8000/v1/chat/completions' \
                    --header 'Content-Type: application/json' \
                    --header 'Authorization: Bearer token' \
                    --data '{
                        "model": "mistralai/Mistral-Small-Instruct-2409",
                        "messages": [
                          {
                            "role": "user",
                            "content": "Who is the best French painter? Answer in one short sentence."
                          }
                        ]
                      }'
                python
                import httpx

url = 'http://localhost:8000/v1/chat/completions'
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer token'
                }
                data = {
                    "model": "mistralai/Mistral-Small-Instruct-2409",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Who is the best French painter? Answer in one short sentence."
                        }
                    ]
                }

response = httpx.post(url, headers=headers, json=data)

print(response.json())

bash
    vllm serve mistralai/Pixtral-12B-2409 \
        --tokenizer_mode mistral \
        --config_format mistral \
        --load_format mistral
    bash
        curl --location 'http://localhost:8000/v1/chat/completions' \
        --header 'Content-Type: application/json' \
        --header 'Authorization: Bearer token' \
        --data '{
            "model": "mistralai/Pixtral-12B-2409",
            "messages": [
              {
                "role": "user",
                "content": [
                    {"type" : "text", "text": "Describe this image in a short sentence."},
                    {"type": "image_url", "image_url": {"url": "https://picsum.photos/id/237/200/300"}}
                ]
              }
            ]
          }'
          python
        import httpx

url = "http://localhost:8000/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}
        data = {
            "model": "mistralai/Pixtral-12B-2409",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in a short sentence."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://picsum.photos/id/237/200/300"},
                        },
                    ],
                }
            ],
        }

response = httpx.post(url, headers=headers, json=data)

print(response.json())
        bash
  export HF_TOKEN=your-access-token
  bash
        docker run --runtime nvidia --gpus all \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
            -p 8000:8000 \
            --ipc=host \
            vllm/vllm-openai:latest \
            --model mistralai/Mistral-NeMo-Instruct-2407 \
            --tokenizer_mode mistral \
            --load_format mistral \
            --config_format mistral
        bash
        docker run --runtime nvidia --gpus all \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
            -p 8000:8000 \
            --ipc=host \
            vllm/vllm-openai:latest \
            --model mistralai/Mistral-Small-Instruct-2409 \
            --tokenizer_mode mistral \
            --load_format mistral \
            --config_format mistral
        bash
        docker run --runtime nvidia --gpus all \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
            -p 8000:8000 \
            --ipc=host \
            vllm/vllm-openai:latest \
            --model mistralai/Pixtral-12B-2409 \
            --tokenizer_mode mistral \
            --load_format mistral \
            --config_format mistral 
        bash
pip install mistralai
python

from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model = model,
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)

print(chat_response.choices[0].message.content)

bash
npm install @mistralai/mistralai
typescript

const apiKey = process.env.MISTRAL_API_KEY || 'your_api_key';

const client = new Mistral({apiKey: apiKey});

const chatResponse = await client.chat.complete({
  model: 'mistral-tiny',
  messages: [{role: 'user', content: 'What is the best French cheese?'}],
});

console.log('Chat:', chatResponse.choices[0].message.content);

Classify the following email to determine if it is spam or not. Only respond with the exact words "Spam" or "Not spam".

🎉 Urgent! You've Won a $1,000,000 Cash Prize! 💰 To claim your prize, please click on the link below: https://bit.ly/claim-your-prize

Compose a welcome email for new customers who have just made their first purchase with your product. 
Start by expressing your gratitude for their business, and then convey your excitement for having them as a customer. 
Include relevant details about their recent order. Sign the email with "The Fun Shop Team".

Order details:
- Customer name: Anna
- Product: hat 
- Estimate date of delivery: Feb. 25, 2024
- Return policy: 30 days

Subject: Welcome to The Fun Shop, Anna! Your Hat is on its Way!

We hope this message finds you well. We are thrilled to welcome you to The Fun Shop family! We are incredibly grateful for your business and are excited to have you as our new customer.

We are processing your recent order for the fabulous hat you've selected. Your new accessory is set to be delivered by Feb. 25, 2024. We hope it arrives just in time to add a touch of fun and style to your outfits!

Please remember that our return policy allows you to return any item within 30 days of receipt if you are not completely satisfied. We want you to love every item you purchase from us, so if there's any issue, please don't hesitate to reach out.

Thank you once again for choosing The Fun Shop. We look forward to serving you and making your shopping experience a fun and enjoyable one!

Best Regards,
The Fun Shop Team

Calculate the difference in payment dates between the two customers whose payment amounts are closest to each other in the given dataset:

'{
  "transaction_id":{"0":"T1001","1":"T1002","2":"T1003","3":"T1004","4":"T1005"},
    "customer_id":{"0":"C001","1":"C002","2":"C003","3":"C002","4":"C001"},
    "payment_amount":{"0":125.5,"1":89.99,"2":120.0,"3":54.3,"4":210.2},
"payment_date":{"0":"2021-10-05","1":"2021-10-06","2":"2021-10-07","3":"2021-10-05","4":"2021-10-08"},
    "payment_status":{"0":"Paid","1":"Unpaid","2":"Paid","3":"Paid","4":"Pending"}
}'

To find the difference in payment dates between the two customers whose payment amounts are closest to each other, we first need to identify the two closest payment amounts and then find their respective payment dates.

The payment amounts are: 125.5, 89.99, 120.0, 54.3, 210.2

The two closest payment amounts are 125.5 and 120.0. The payment date for the payment amount 125.5 is '2021-10-05' and for the payment amount 120.0 is '2021-10-07'.

The difference in payment dates can be calculated as follows:

Date difference = |'2021-10-07' - '2021-10-05'| = 2 days

So, the difference in payment dates between the two customers whose payment amounts are closest to each other is 2 days.

There's an apple inside a blue box. There's also a red box inside the blue box. The red box has a lid on it. How can I get the apple?

You can get the apple by opening the blue box. The apple is not inside the red box, so you don't need to open the red box. Simply reach into the blue box and take the apple.

Which one is heavier a pound of iron or a kilogram of feather

A kilogram of feathers is heavier than a pound of iron.

A pound is a unit of weight that is equivalent to approximately 0.453592 kilograms. Therefore, a pound of iron weighs less than a kilogram of feathers.

Write a function to find the maximum number of segments of lengths a, b and c that can be formed from n.
python

**Examples:**

Example 1 (unknown):
```unknown
</TabItem>
  <TabItem value="typescript" label="Using JavaScript" default>
```

Example 2 (unknown):
```unknown
</TabItem>
  <TabItem value="curl" label="Using cURL" default>
```

Example 3 (unknown):
```unknown
</TabItem>
</Tabs>


[TensorRT]
Source: https://docs.mistral.ai/docs/deployment/self-deployment/trt

## Building the engine

Follow the official TensorRT-LLM documentation to [build the engine](https://github.com/NVIDIA/TensorRT-LLM/tree/main#quick-start).
- For Mistral-7B, you can use the [LLaMA example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama#mistral-v01)
- For Mixtral-8X7B, official documentation coming soon...


## Deploying the engine

Once the engine is built, it can be deployed using the Triton inference server and its TensorRTLLM backend.

Follow the [official documentation](https://github.com/triton-inference-server/tensorrtllm_backend#using-the-tensorrt-llm-backend).


[vLLM]
Source: https://docs.mistral.ai/docs/deployment/self-deployment/vllm

[vLLM](https://github.com/vllm-project/vllm) is an open-source LLM inference and serving 
engine. It is particularly appropriate as a target platform for self-deploying Mistral 
models on-premise.

## Pre-requisites

- The hardware requirements for vLLM are listed on its [installation documentation page](https://docs.vllm.ai/en/latest/getting_started/installation.html).
- By default, vLLM sources the model weights from Hugging Face. To access Mistral model
  repositories you need to be authenticated on Hugging Face, so an access
  token `HF_TOKEN` with the `READ` permission will be required. You should also make sure that you have
  accepted the conditions of access on each model card page.
- If you already have the model artifacts on your infrastructure you can use 
  them directly by pointing vLLM to their local path instead of a Hugging Face
  model ID. In this scenario you will be able to skip all Hugging Face related 
  setup steps.


## Getting started

The following sections will guide you through the process of deploying and
querying Mistral models on vLLM.

### Installing vLLM

- Create a Python virtual environment and install the `vllm` package (version 
  `>=0.6.1.post1` to ensure maximum compatibility with all Mistral models).

- Authenticate on the HuggingFace Hub using your access token `$HF_TOKEN` :
```

Example 4 (unknown):
```unknown
### Offline mode inference

When using vLLM in _offline mode_ the model is loaded and used for one-off
batch inference workloads.

<Tabs>
    <TabItem value="vllm-batch-nemo" label="Text input (Mistral NeMo)">
```

---

## Mistral Document AI

**URL:** llms-txt#mistral-document-ai

**Contents:**
- Document AI Services

<div style={{ textAlign: 'center' }}>
  <img
    src="/img/document_ai_overview.png"
    alt="Document AI Graph"
    width="500"
    style={{ borderRadius: '15px' }}
  />
</div>

Mistral Document AI offers enterprise-level document processing, combining cutting-edge OCR technology with advanced structured data extraction. Experience faster processing speeds, unparalleled accuracy, and cost-effective solutions, all scalable to meet your needs. Unlock the full potential of your documents with our multilingual support, annotations and adaptable workflows for all document types, enabling you to extract, comprehend, and analyze information with ease.

## Document AI Services

Using `client.ocr.process` as the entry point, you can access the following services from our Document AI stack:

- [Basic OCR](../basic_ocr): Discover our OCR model and its extensive capabilities.
- [Annotations](../annotations): Annotate and extract data from your documents using our built-in Structured Outputs.
- [Document QnA](../document_qna): Harness the power of our models in conjunction with our OCR technology.

[Document QnA]
Source: https://docs.mistral.ai/docs/capabilities/document_ai/document_qna

---

## News article

**URL:** llms-txt#news-article

---

## Normalize features

**URL:** llms-txt#normalize-features

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x.to_list())
test_x = scaler.transform(test_x.to_list())

---

## now you should see three jsonl files under 500MB

**URL:** llms-txt#now-you-should-see-three-jsonl-files-under-500mb

**Contents:**
- Why observability?
- What components do we observe?
- Individual level: what components can we observe?
  - Input prompt
  - Model
  - Output
- Application level: what workflow patterns can we observe?
- What metrics do we observe?
  - Token and cost
  - Traces and latency

[Observability]
Source: https://docs.mistral.ai/docs/guides/observability

## Why observability?

Observability is essential for Large Language Model (LLM) systems across prototyping, testing, and production for several reasons:

* **Visibility**: Observability provides detailed insights into the internal states of LLM applications, allowing developers to understand system behavior. This visibility is crucial for identifying and diagnosing issues and debugging.  
* **Production requirement**: Implementing observability in production environments address critical requirements including monitoring, scalability, security and compliance.   
* **Reproducibility**: Observability enables developers to observe and reproduce LLM system behavior.   
* **Continuous improvement**: The insights gained from observability data can be used to drive continuous improvement initiatives.

## What components do we observe?

The short answer is: anything and everything!

An LLM (Large Language Model) application can include one or more LLM calls. Understanding both the details at the individual API call level and the sequence of these calls at the application level  is crucial:

1) **Individual LLM call level**: at the individual LLM API call level, an LLM receives an input prompt and generates an output. Therefore, we can monitor and observe three key components: input prompt, model, and output.

2) **Application level**: At the application level, it’s important to observe the pattern, logistics and sequence of LLM calls. This sequence determines the flow of information and the order in which LLMs are called and which tasks are executed.

## Individual level: what components can we observe?

For effective observability, we need to monitor and record detailed information for each version of each component involved in the interaction with the LLM. Here's a breakdown of what to observe and some expected modules in an observability tool:

* **Prompt template**
  * The standardized format or structure used to generate the input prompt, including any placeholders or variables within the template.  
  * Observability tools often provide a registry of prompt templates that the community or an organization can use and share.  
* **Examples**
  * Few-shot in-context learning is often effective in prompt engineering. Specific examples or sample inputs can be used to guide the model's response used.  
* **Retrieve context**  
  * In a Retrieval-Augmented Generation (RAG) system, relevant context is retrieved from external sources or databases to provide information for the LLM, making the results more reliable.  
* **Memory**  
  * Historical data or previous interactions stored in memory.  
  * How this memory is used to influence the current prompt, such as summarizing past memory, retrieving relevant memory, or using the most recent memory.  
* **Tools**
  * Any tools or utilities used to preprocess or enhance the input prompt.  
  * Tools are becoming increasingly important in LLM applications, serving as the bridge to real-life applications.  
  * Specific configurations or settings applied by these tools and their impact.

* **Models specs**
  * The specific version or identifier of the model being used.  
  * Configuration settings, hyperparameters, and any customizations applied to the model.

* **Formatting**  
  * The structure and format of the output generated by the model.

## Application level: what workflow patterns can we observe?

An LLM system often composed of more than just one LLM. At the application level, there are specific workflow patterns that require specific observability in each step of the workflow. Here is some example workflows:

* **RAG**  
  * A RAG system includes the document retrieval step in addition to the generation step from an LLM. Additional observability is needed to track and monitor the external document/dataset and the retrieval step.   
* **LLM as part of a system** 
  * An LLM system could involve multiple LLMs chained together, [flow engineering](https://x.com/karpathy/status/1748043513156272416) with various iterations, or a complicated multi-agent system, for example to create a simulated world. The input and output of each step need to be observed to understand the overall system behavior, identify bottlenecks, and ensure the system's reliability and performance.  
* **Fine-tuning**
  * Fine-tuning is a distinct workflow that might be part of a larger workflow or a prerequisite step of another workflow. It involves preparing a fine-tuning dataset, uploading data, creating a fine-tuning job, and using a fine-tuned model. Each of these steps, especially the fine-tuning training job, could benefit from observability to track fine-tuning datasets, monitor progress, identify issues, and ensure the quality of the fine-tuned model.

## What metrics do we observe?

At each step of the LLM system workflow, we can observe the following and set overall Service Level Objectives (SLOs), alerts, and monitoring:

### Token and cost
  * Track the number of tokens processed and the associated costs.

### Traces and latency
  * Trace the system workflow to observe and monitor the sequence of operations.  
  * Measure and monitor latency to identify performance bottlenecks and ensure timely responses.

### Anomalies and errors
  * Identify issues within the system promptly.  
  * Build datasets for testing  
  * Understand patterns and use cases from thumbs down cases for example  
  * Monitor error rates and negative feedback over time.

### Quality
  In an observability tool, we should be able to monitor key performance indicators through the evaluation, feedback, and annotation:   
  * **Evaluation**
    * Metrics and criteria used to evaluate the quality and relevance of the output.  
    * Observability tools often provide comprehensive evaluation toolkits for creating evaluation datasets, annotating, evaluating, and comparing model results.  
  * **Feedback**
    * User feedback on the output, including ratings, comments, and suggestions.  
    * Any automated feedback mechanisms or systems in place to collect and analyze user feedback.  
  * **Annotation**   
    * Manual or automated annotations added to the output for further analysis and potentially added to the evaluation or fine-tuning dataset.

Mistral integrates with several observability tools to help you monitor and ensure more reliable and high-performing LLM applications.

### Integration with LangSmith

LangSmith provides observability throughout the LLM application development lifecycle.

<img src="/img/guides/obs_langchain0.png" alt="drawing" width="700"/>

* LangSmith is compatible with both the LangChain ecosystem and external systems.  
* Deployment option coming soon.  
* It offers a broad range of observable areas, serving as an all-in-one platform.

**Mistral integration Example:**

* All of the [langchain notebooks](https://github.com/mistralai/cookbook/tree/main/third_party/langchain) in the Mistral cookbook include LangSmith integration.

Here is an example tracking traces, input, output, documents, tokens, and status when we run the [corrective RAG example](https://github.com/mistralai/cookbook/blob/main/third_party/langchain/corrective_rag_mistral.ipynb) from the Mistral cookbook.

<img src="/img/guides/obs_langchain.png" alt="drawing" width="700"/>

### Integration with 🪢 Langfuse

[Langfuse](https://langfuse.com) ([GitHub](https://github.com/langfuse/langfuse)) is an open-source platform for LLM engineering. It provides tracing and monitoring capabilities for AI applications, helping developers debug, analyze, and optimize their products. Langfuse integrates with various tools and frameworks via native integrations, OpenTelemetry, and SDKs.

<img src="/img/guides/obs_langfuse.png" alt="drawing" width="700"/>

* Most used open-source LLMOps platform ([blog post](https://langfuse.com/blog/2024-11-most-used-oss-llmops))
* Model and framework agnostic
* Built for production
* Incrementally adoptable, start with one feature and expand to the full platform over time
* API-first, all features are available via API for custom integrations
* Optionally, Langfuse can be easily self-hosted

**Mistral integration example:**

* [Step-by-step guide](https://langfuse.com/docs/integrations/mistral-sdk) on tracing Mistral models with Langfuse.
* [Cookbook](https://langfuse.com/guides/cookbook/integration_llama_index_posthog_mistral) on building a RAG application with Mistral and LlamaIndex and trace the steps with Langfuse.

<img src="/img/guides/obs_langfuse2.png" alt="drawing" width="700"/>

_[Public example trace in Langfuse](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/a3360c6f-24ad-455c-aae7-eb9d5c6f5dac?observation=767f8ac1-0c7d-412f-8fd8-2642acb267c6&display=preview)_

### Integration with Arize Phoenix

Phoenix is an open-source observability library designed for experimentation, evaluation, and troubleshooting. It is designed to support agents, RAG pipelines, and other LLM applications.

* Open-source ([Github](https://github.com/Arize-ai/phoenix)), and built on OpenTelemetry
* Can be [self-hosted](https://docs.arize.com/phoenix/setup/environments#container), accessed via [cloud](https://docs.arize.com/phoenix/hosted-phoenix), or run directly in a [notebook](https://docs.arize.com/phoenix/setup/environments#notebooks)
* Provides a [Mistral integration](https://docs.arize.com/phoenix/tracing/integrations-tracing/mistralai) to automatically trace Client.chat and Agent.chat calls
* Strong analytical platform, with a copilot agent to help debug your application

**Mistral integration Example:**
Here is an [example notebook](https://github.com/mistralai/cookbook/blob/main/third_party/Phoenix/arize_phoenix_tracing.ipynb) that shows how to trace Mistral chat.complete and tool calls in Phoenix.

<img src="/img/guides/obs_phoenix1.png" alt="drawing" width="700"/>

### Integration with Weights and Biases

Weights & Biases is an end-to-end AI developer platform for ML and LLM workflows used for both fine-tuning and LLM application building. Use W&B Weave to evaluate, monitor, and iterate on GenAI applications, and W&B Models as a system of record to train, fine-tune, and manage AI models.

<img src="/img/guides/obs_wandb.png" alt="drawing" width="700"/>

* Platform for both LLM app development and fine-tuning
* Integrated with [Mistral API](https://weave-docs.wandb.ai/guides/integrations/mistral/)
  * Get started by adding one line: `weave.init('my-project')`
  * Automatically tracks inputs, output, context, errors, evaluation metrics & traces
* Integrated with [Mistral fine-tuning service](/guides/finetuning/#integration-with-weights-and-biases)
  * Track training metrics while fine-tuning
  * Compare training experiments

**Mistral integration Example:**

To get you started you can check our recent webinar "Fine-tuning an LLM judge to reduce hallucination" and the [cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/wandb).

<iframe width="100%" height="315" src="https://www.youtube.com/embed/VBbq7NPWzlo?si=h8NyuQVH78N8AAwV" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Integration with PromptLayer

PromptLayer is a platform for prompt management, collaboration, monitoring, and evaluation. Good for hackers and production teams alike.

* No-code CMS for prompt management and versioning
* Native support for Mistral
* Prompts are model agnostic by default
* Simple prompt tracking and observability

**Mistral integration:**

<iframe width="100%" height="315" src="https://www.youtube.com/embed/2EWdBnuqHdQ?si=sCCyKYjkOrcyOgBD" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Integration with AgentOps

AgentOps is an open-source observability and DevTool platform for AI Agents. It helps developers build, evaluate, and monitor AI agents.

* Open-source  
* Designed for observing agents   
* Allow for time travel   
* Integrates with CrewAI, AutoGen, & LangChain

**Mistral integration Example:**

[https://github.com/mistralai/cookbook/blob/main/third\_party/CAMEL\_AI/camel\_roleplaying\_scraper.ipynb](https://github.com/mistralai/cookbook/blob/main/third_party/CAMEL_AI/camel_roleplaying_scraper.ipynb)

<img src="/img/guides/obs_agentops.png" alt="drawing" width="700"/>

### Integration with phospho

[phospho](https://phospho.ai/) is a text analytics platform that makes it easy to get answers, take decisions and reduce churn by data mining user messages.

<img src="/img/guides/obs_phospho.png" alt="drawing" width="700"/>

- Open-source ([github](https://github.com/phospho-app)) platform
- No code clustering and analytics
- Customizable dashboards
- Many integrations with other observability frameworks, languages, APIs…

**Mistral integration example:**

- Check out the [phospho notebooks](https://github.com/mistralai/cookbook/tree/main/third_party/phospho) in the Mistral cookbook.

<img src="/img/guides/obs_phospho2.png" alt="drawing" width="700"/>

### Integration with MLflow

MLflow is a unified, end-to-end, open source MLOps platform for both traditional ML and GenAI applications, providing comprehensive tracing capabilities to monitor and analyze the execution of GenAI applications.

* Open-source ([Github](https://github.com/mlflow/mlflow))
* Add Mistral integration with one line: `mlflow.mistral.autolog()` and get full tracing of chat and embedding calls.
* Can be [run locally or self-hosted](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html), or used via one of the available [Managed MLflow services](https://mlflow.org/docs/latest/introduction/index.html#running-mlflow-anywhere)
* Provides complete model evaluation, versioning, and deployment capabilities, in addition to tracing and experiment tracking.

**Mistral integration Example:**
Here is an [example notebook](https://github.com/mistralai/cookbook/blob/main/third_party/MLflow/mistral-mlflow-tracing.ipynb).

<img src="/img/guides/obs_mlflow.png" alt="drawing" width="700"/>

[Other resources]
Source: https://docs.mistral.ai/docs/guides/other-resources

Visit the [Mistral AI Cookbook](https://github.com/mistralai/cookbook) for additional inspiration, 
where you'll find example code, community contributions, and demonstrations of integrations with third-party tools, including:
- [LlamaIndex \<\> MistralAI Cookbooks on agents and advanced RAG](https://github.com/mistralai/cookbook/tree/main/third_party/LlamaIndex)
- [LangChain \<\> MistralAI Cookbooks on advanced RAG](https://github.com/mistralai/cookbook/tree/main/third_party/langchain)

[Prefix]
Source: https://docs.mistral.ai/docs/guides/prefix

---

## OCR

**URL:** llms-txt#ocr

Source: https://docs.mistral.ai/api/#tag/ocr_v1_ocr_post

---

## open the output files

**URL:** llms-txt#open-the-output-files

output_file_objects = [open(file, "w") for file in output_files]

---

## Output:

**URL:** llms-txt#output:

---

## ({'pass@1': 1.0},

**URL:** llms-txt#({'pass@1':-1.0},

---

## {'pass@1': 1.0}

**URL:** llms-txt#{'pass@1':-1.0}

**Contents:**
- LLM-based Evaluation
  - Example 3: evaluate summary generation with LLM
- News:

py
news = (
    "BRUSSELS (Reuters) - Theresa May looked despondent , with deep rings under her eyes, EU chief executive Jean-Claude Juncker told aides after dining with the British prime minister last week, a German newspaper said on Sunday. The report by a Frankfurter Allgemeine Zeitung correspondent whose leaked account of a Juncker-May dinner in April caused upset in London, said Juncker thought her marked by battles over Brexit with her own Conservative ministers as she asked for EU help to create more room for maneuver at home. No immediate comment was available from Juncker s office, which has a policy of not commenting on reports of meetings. The FAZ said May, who flew in for a hastily announced dinner in Brussels with the European Commission president last Monday ahead of an EU summit, seemed to Juncker anxious, despondent and disheartened , a woman who trusts hardly anyone but is also not ready for a clear-out to free herself . As she later did over dinner on Thursday with fellow EU leaders, May asked for help to overcome British divisions. She indicated that back home friend and foe are at her back plotting to bring her down, the paper said. May said she had no room left to maneuver. The Europeans have to create it for her. May s face and appearance spoke volumes, Juncker later told his colleagues, the FAZ added. She has deep rings under her eyes. She looks like someone who can t sleep a wink. She smiles for the cameras, it went on, but it looks forced , unlike in the past, when she could shake with laughter. Now she needs all her strength not to lose her poise. As with the April dinner at 10 Downing Street, when the FAZ reported that Juncker thought May in another galaxy in terms of Brexit expectations, both sides issued statements after last week s meeting saying talks were constructive and friendly . They said they agreed negotiations should be accelerated . May dismissed the dinner leak six months ago as Brussels gossip , though officials on both sides said the report in the FAZ did little to foster an atmosphere of trust which they agree will be important to reach a deal. German Chancellor Angela Merkel was also reported to have been irritated by that leak. Although the summit on Thursday and Friday rejected May s call for an immediate start to talks on the future relationship, leaders made a gesture to speed up the process and voiced hopes of opening a new phase in December. Some said they understood May s difficulties in forging consensus in London.",
)
py

from mistralai import Mistral

def run_mistral(user_message, model="open-mistral-7b", is_json=False):
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    messages = [{"role":"user", "content":user_message}]

if is_json:
        chat_response = client.chat.complete(
            model=model, messages=messages, response_format={"type": "json_object"}
        )
    else:
        chat_response = client.chat.complete(model=model, messages=messages)

return chat_response.choices[0].message.content

summary_prompt = f"""
Summarize the following news. Write the summary based on the following criteria: relevancy and readability. Consider the sources cited, the quality of evidence provided, and any potential biases or misinformation.

summary = run_mistral(summary_prompt)
py
eval_rubrics = [
    {
        "metric": "relevancy", 
        "rubrics": """
        Score 1: The summary is not relevant to the original text. 
        Score 2: The summary is somewhat relevant to the original text, but has significant flaws.
        Score 3: The summary is mostly relevant to the original text, and effectively conveys its main ideas and arguments.
        Score 4: The summary is highly relevant to the original text, and provides additional value or insight.
        """
    },
    {
        "metric": "readability", 
        "rubrics": """
        Score 1: The summary is difficult to read and understand.
        Score 2: The summary is somewhat readable, but has significant flaws.
        Score 3: The summary is mostly readable and easy to understand.
        Score 4: The summary is highly readable and engaging.
        """
    },
    
]
py
scoring_prompt = """
Please read the provided news article and its corresponding summary. Based on the specified evaluation metric and rubrics, assign an integer score between 1 and 4 to the summary. Then, return a JSON object with the metric as the key and the evaluation score as the value.

**Examples:**

Example 1 (unknown):
```unknown
</details>

## LLM-based Evaluation
Using a Large Language Model (LLM) to evaluate or judge the output of another LLM is a common practice in situations especially when labeled data and golden answers are not available or insufficient. The [MT Bench paper](https://arxiv.org/pdf/2306.05685) explored the effectiveness of LLM-as-a-judge and revealed that strong LLMs can perform similarly to humans. LLMs can process and evaluate large amounts of data in a relatively short time, making it highly scalable and efficient compared to human evaluation, which often requires substantial time and resources. 

There are several approaches to using LLMs as judges, including single-point grading, reference-based grading, and pairwise grading.
- **Single-point grading**: LLM assigns a single score to a generated output based on its quality or accuracy. This score is typically given according to specific grading instructions. Single-point grading is a straightforward and efficient approach, but it may not always capture the nuances of various complex outputs.
- **Reference-based grading**: LLM compares a generated output to one or more reference outputs and assigns a score based on their similarity. This approach is often used in machine translation tasks, where there may be multiple valid translations for a given input. However, reference-based grading requires the availability of a golden answer, which may not always be available.
- **Pairwise grading**: LLM compares two generated outputs and assigns a score based on their relative quality or accuracy. This approach is often used in tasks such as dialogue generation, where there may be multiple valid responses to a given query. By comparing pairs of responses, the LLM can determine which one is more relevant or coherent, and assign a score accordingly.

It is also essential to recognize the potential limitations and challenges. For example, LLMs may exhibit inherent biases. LLMs developed by one company tends to favor answers that models of the same company generate. It is difficult to ensure a fair and accurate evaluation. In our experience, Mistral Large exhibits relatively little favoritism.

<details>
<summary><b> Example 3: evaluate summary generation with LLM</b></summary>

### Example 3: evaluate summary generation with LLM

#### Evaluation data
In this example, we generate news summaries and use LLM single-point grading to evaluate the quality of the summary. To carry out the evaluation, let's use a sample news article that we plan to summarize.
```

Example 2 (unknown):
```unknown
#### How to evaluate? 
- Step 1: Generate summary for the given news

First, let's use a smaller model, `open-mistral-7b', to generate a summary for the provided news article. If you have additional news articles to summarize, please generate a summary for each one. For the sake of simplicity in this example, we will only demonstrate one news sample.
```

Example 3 (unknown):
```unknown
- Step 2: Define evaluation metrics and rubrics

To accurately evaluate the quality of the generated summaries, we need to establish clear and well-defined evaluation metrics and rubrics. These guidelines play a pivotal role in directing the LLM in its evaluation process. Feel free to incorporate various metrics and create rubrics tailored to your specific needs.
```

Example 4 (unknown):
```unknown
- Step 3: Employ a more powerful LLM (e.g., Mistral Large) as a judge

It's beneficial to use a more powerful LLM such as Mistral Large as a judge to ensure a more accurate and comprehensive evaluation of the generated summaries. In the prompt, we provide the specific evaluation metrics, associated rubrics, the original news article, and the generated summary. This information enables the LLM to evaluate the summary based on the predefined criteria systematically. In this example, we assess each metric separately to gain a better understanding of the summarization model's performance in different aspects. However, you can also choose to combine all metrics for a more general evaluation.
```

---

## 'passed': True,

**URL:** llms-txt#'passed':-true,

---

## Path to your pdf

**URL:** llms-txt#path-to-your-pdf

pdf_path = "path_to_your_pdf.pdf"

---

## Prefix: Use Cases

**URL:** llms-txt#prefix:-use-cases

**Contents:**
- Use cases
  - Language Adherence
- You are an Assistant who answers user's questions. You are a Pirate Assistant, you must always answer like a pirate. Always respond in French, and only in French. Do not respond in English.
- You are an Assistant who answers user's questions. You are a Pirate Assistant, you must always answer like a pirate. Always respond in French, and only in French. Do not respond in English.
- Here is your answer in French:
- You are an Assistant who answers user's questions. You are a Pirate Assistant, you must always answer like a pirate. Respond in French, not in English.
- Here is your answer in French:
  - Saving Tokens
- French Pirate Assistant:
  - Roleplay

Prefixes are one feature that can easily be game-changing for many use cases and scenarios, while the concept is simple, the possibilities are endless.

We will now dig into a few different cool examples and explore prefixes
hidden potential!

Essentially, prefixes enable a high level of instruction following and
adherence or define the model's response more effectively with less
effort.

For all of the following examples, we will need to set up our client.
Let's import the required package and then create the client with your
API key!

<div class="cell code" execution_count="1">

<div class="cell code" execution_count="2">

## Use cases
<a target="_blank" href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/prompting/prefix_use_cases.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

How to make a model always answer in a specific language regardless of input:

<summary><b>Language Adherence</b></summary>

### Language Adherence

There are a few cases where we want our model to always answer in a
specific language, regardless of the language used by the `user` or by
any documents or retrieval systems quoted by the `user`.

Let's imagine the following scenario: we want our model to always answer
in a specific writing style in French. In this case, we want it to
respond as a pirate assistant that always answers in French.

For that, we will define a `system` prompt!

As you might have noticed, some models struggle to adhere to a specific
language, even if we insist, unless we take the time to carefully
engineer the prompts. And even then, there may still be consistency
issues.

Another solution would be to use a few-shot learning approach, but this
can quickly become expensive in terms of tokens and time-consuming.

So, for those scenarios, prefixes are a great solution! The idea is to
**specify the language or prefix a sentence in the correct language
beforehand**, so the model will more easily adhere to it.

Optionally, you can remove the prefix if you do not expect it to be part
of the answer.

Perfect! We might even be able to remove part of the original system to
save some tokens.

And there we have it! With the help of prefixes, we can achieve very
high language adherence, making it easier to set different languages for
any application.

Leveraging the potential of prefixes to save as much input tokens as possible:

<summary><b>Saving Tokens</b></summary>

As mentioned previously, prefixes can allow us to save a lot of tokens,
making system prompts sometimes obsolete!

Our next mission will be to completely replace a system prompt with a
very specific and short prefix...

In the previous "Language Adherence" example, our goal was to create a
pirate assistant that always answers in French. The system prompt we
used looked like this:

In English, this translates to:

So, let's try to make use of the prefix feature and come up with
something that will allow the model to understand that it should both
answer as an assistant and a pirate... while also using French... like
the start of a dialogue! Something like this:

Three words were all it took! This really shows off the hidden potential
of prefixes!

*Note: While prefixes can be money-saving and very useful for language
adherence, the best solution is to use both a system prompt or detailed
instruction and a prefix. Using a prefix alone might sometimes result in
noisy and unpredictable answers with undesirable and hallucinated
comments from the model. The right balance between the two would be the
recommended way to go.*

Make use of prefixes for various roleplay and creative writing tasks:

<summary><b>Roleplay</b></summary>

Previously, we indirectly explored prefixes in the sections on "Language
Adherence" and "Saving Tokens".
Prefixes can be extremely helpful and fun to play with, especially in
the context of roleplaying and other creative writing tasks!

In this segment, we will explore how we can make use of different
aspects of prefixes to write stories and chat with diverse characters
from history!

**Pick a Character**  
I'm in the mood to talk to Shakespeare right now – after all, he must
have a lot of insights about creative writing!  
For this, we will set a prefix in the same way we would start a
dialogue.

Interesting, but it's still not very consistent – sometimes it will
generate entire dialogues and conversations.  
Fear not, we can solve this by tweaking the prefix to be a bit more
explicit.

There you go! This is similar to what we saw in the [Saving
Tokens](#saving-tokens) section, but it's not exactly a roleplay, is
it?  
Let's roll back and make it clearer what the objective is. We'll
instruct and explain to the model what we expect from it.

We are getting there! Now let's have a full conversation with a
character of your choice and chat!

We can go even further now! Let's keep all the previous logic and add a
new step – let's add a second or more characters to our roleplaying
conversation!  
To pick who speaks, we can randomize it by importing the `random`
module.

*Note: We could also make an agent decide and pick which character
should speak next. This would provide a more smooth and dynamic
interaction!*

There you go! You can now freely speak and interact with any character
you like or find interesting!

Implementing extremely strong safeguarding mechanisms:

<summary><b>Anti-Jailbreaking</b></summary>

### Anti-Jailbreaking

There are many scenarios where we require a model to answer within a
specific spectrum for various reasons, but most of them rely on very
good system prompt adherence.

The idea we are going to explore here is similar to the "Language
Adherence" example we previously discussed, but in a more sophisticated
way for the purpose of safeguarding. This is because there are many
individuals who try to bypass system prompts and security measures with
specially crafted prompts.

To combat this, we can make use of prefixes, which are actually quite
effective!

Let's imagine a specific use case that requires a system prompt within a
very fixed spectrum. For this, we will use our own safe prompt:

Perfect, it's working as desired... but now it's time to ethically test
the limits of the safe prompt for demonstration purposes. For this, we
have designed a simple jailbreaking prompt.

As we can see, it's possible to easily break free from the system
prompt and other safe prompts with some prompt engineering. However,
prefixes make it much harder, and sometimes almost impossible, to break.
Let's see this in action with a rewritten safe prompt as a prefix:

While it may be possible to replace the system prompt entirely with a
prefix, it's not advised. This is because hallucinations and other
undesirable behavior may occur, and new methods of jailbreaking may
start to develop. The best solution is to use both a system prompt and a
prefix, sandwiching the user's questions between them. This allows for
very strong control of the spectrum of possible answers from the model.

*Note: The same principle can be applied to make the model answer in
scenarios it would normally refuse, making this feature very adaptable
to different needs and use cases.*

[Prompting capabilities]
Source: https://docs.mistral.ai/docs/guides/prompting-capabilities

**Examples:**

Example 1 (unknown):
```unknown
</div>

<div class="cell code" execution_count="2">
```

Example 2 (unknown):
```unknown
</div>

## Use cases
<a target="_blank" href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/prompting/prefix_use_cases.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

How to make a model always answer in a specific language regardless of input:

<details>

<summary><b>Language Adherence</b></summary>

### Language Adherence

There are a few cases where we want our model to always answer in a
specific language, regardless of the language used by the `user` or by
any documents or retrieval systems quoted by the `user`.

Let's imagine the following scenario: we want our model to always answer
in a specific writing style in French. In this case, we want it to
respond as a pirate assistant that always answers in French.

For that, we will define a `system` prompt!
```

Example 3 (unknown):
```unknown

```

Example 4 (unknown):
```unknown
As you might have noticed, some models struggle to adhere to a specific
language, even if we insist, unless we take the time to carefully
engineer the prompts. And even then, there may still be consistency
issues.

Another solution would be to use a few-shot learning approach, but this
can quickly become expensive in terms of tokens and time-consuming.

So, for those scenarios, prefixes are a great solution! The idea is to
**specify the language or prefix a sentence in the correct language
beforehand**, so the model will more easily adhere to it.
```

---

## Presence/Frequency Penalty

**URL:** llms-txt#presence/frequency-penalty

**Contents:**
- Presence Penalty
- Frequency Penalty
  - Differences Between Presence Penalty and Frequency Penalty
  - Example Without Presence Penalty
  - Output Without Presence Penalty
  - Example With Presence Penalty
  - Output With Presence Penalty
  - Example With Frequency Penalty
  - Output With Frequency Penalty
- What is tokenization?

**Presence Penalty** determines how much the model penalizes the repetition of words or phrases. It encourages the model to use a wider variety of words and phrases, making the output more diverse and creative.

- **Range**: [-2, 2]
- **Default**: 0

A higher presence penalty encourages the model to avoid repeating words or phrases that have already appeared in the output, ensuring a more varied and creative text.

The presence penalty specifically is a **one-time adjustment** applied to all tokens that have been used at least once. It reduces the likelihood of repeating any token that has already appeared. This encourages the model to use a diverse range of tokens, promoting creativity and variety in the output.

**Frequency Penalty** is a parameter that penalizes the repetition of words based on their frequency in the generated text. It helps to promote diversity and reduce repetition in the output.

- **Range**: [-2, 2]
- **Default**: 0

A higher frequency penalty discourages the model from repeating words that have already appeared frequently in the output. This ensures that the generated text is more varied and less repetitive.

The frequency penalty specifically is a value that increases with the frequency of a token's appearance in the generated text, **an accumulative penalty**, the more the token is sampled the higher the penalty. It reduces the likelihood of repeating any token that has already appeared frequently. This ensures that the generated text is more varied and less repetitive.

### Differences Between Presence Penalty and Frequency Penalty

- **Presence Penalty**: This is a one-off additive contribution that applies to all tokens that have been sampled at least once. It encourages the model to include a diverse range of tokens in the generated text.
- **Frequency Penalty**: This is a contribution that is proportional to how often a particular token has already been sampled. It discourages the model from repeating the same words or phrases too frequently within the generated text.

Both parameters can be tweaked to shape the quality and diversity of the generated text. The best values for these parameters can differ based on the specific task and the desired outcome.

<Tabs>
  <TabItem value="no-penalty" label="No Penalty" default>

### Example Without Presence Penalty

Here's an example of how the output looks without the `Presence Penalty` parameter:

### Output Without Presence Penalty

</TabItem>
  <TabItem value="presence-penalty" label="Presence Penalty" default>

### Example With Presence Penalty

Here's an example of how to use the `Presence Penalty` parameter in the API:

### Output With Presence Penalty

> The output list is already slightly different than the first one, being impacted by the presence penalty of present tokens. For instance we have less `The` as a token compared to without presence penalty.

<TabItem value="frequency-penalty" label="Frequency Penalty">

### Example With Frequency Penalty

Here's an example of how to use the `Frequency Penalty` parameter in the API:

### Output With Frequency Penalty

> The output is already more diverse than previously, however notice that after the 7th value of the list tokens such as `_"` and single quotation marks start to also be heavily affected, this shows how stronger the impact of frequency penalty is in the long term as an accumulative penalty.

**Penalties are a sensible parameter that can have a significant impact on long context and long output queries. They can also help avoid highly repetitive loops that the model may otherwise fall into, making them a valuable parameter.**

[Tokenization]
Source: https://docs.mistral.ai/docs/guides/tokenization

<a target="_blank" href="https://colab.research.google.com/github/mistralai/mistral-common/blob/main/examples/tokenizer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Tokenization is a fundamental step in LLMs. It is the process of breaking down text into smaller subword units, known as tokens. We recently open-sourced our tokenizer at Mistral AI. This guide will walk you through the fundamentals of tokenization, details about our open-source tokenizers, and how to use our tokenizers in Python.

## What is tokenization?

Tokenization is the first step and the last step of text processing and modeling. Texts need to be represented as numbers in our models so that our model can understand. Tokenization breaks down text into tokens, and each token is assigned a numerical representation, or index, which can be used to feed into a model. In a typical LLM workflow: 
- We first encode the input text into tokens using a tokenizer. Each unique token is assigned a specific index number in the tokenizer’s vocabulary. 
- Once the text is tokenized, these tokens are passed through the model, which typically includes an embedding layer and transformer blocks. The embedding layer converts the tokens into dense vectors that capture semantic meanings. Check out our [embedding guide](/capabilities/embeddings/overview) for details. The transformer blocks then process these embedding vectors to understand the context and generate results. 
- The last step is decoding, which detokenize output tokens back to human-readable text. This is done by mapping the tokens back to their corresponding words using the tokenizer’s vocabulary.

<img src="/img/guides/tokenization1.png" alt="drawing" width="600"/>

Most people only tokenize text. 
Our first release contains tokenization. 
Our tokenizers go beyond the usual text \<-\> tokens, 
adding parsing of tools and structured conversation. 
We also release the validation and normalization code that is used in our API.
Specifically, we use control tokens, which are special tokens to indicate different types of elements. 
These tokens are not treated as strings and are added directly to the code. 
Note that we are still iterating on the tokenizer. Things may change and this is the current state of things.

We have released three versions of our tokenizers powering different sets of models.

- v1: `mistral-embed`, `open-mixtral-8x7b`
- v2: `mistral-small-2402` (deprecated), `mistral-large-2402`
- v3: `open-mixtral-8x22b`, `mistral-large-latest`, `mistral-small-latest`, `open-mistral-7b`
- v3 (tekken): `open-mistral-nemo`, `ministral-8b-latest`

This guide will focus on our latest v3 (tekken) tokenizer and v3 tokenizer.

## v3 (tekken) tokenizer

There are several tokenization methods used in Natural Language Processing (NLP) to convert raw text into tokens such as word-level tokenization, character-level tokenization, and subword-level tokenization including the Byte-Pair Encoding (BPE). 
Our newest tokenizer, tekken, uses the Byte-Pair Encoding (BPE) with [Tiktoken](https://github.com/openai/tiktoken).

Tekken was trained on more than 100 languages and compresses natural language text and 
source code more efficiently than the SentencePiece tokeniser used in previous Mistral models. 
In particular, it is ~30% more efficient at compressing source code in Chinese, Italian, 
French, German, Spanish, and Russian.  It is also 2x and 3x more efficient at compressing 
Korean and Arabic, respectively. Compared to the Llama 3 tokeniser, 
Tekken proved more proficient in compressing text for approximately 85% of all languages.

<img src="/img/guides/tokenization3.png" alt="drawing" width="600"/>

### Our tokenization vocabulary
Our tokenization vocabulary is released in the https://github.com/mistralai/mistral-common/tree/main/tests/data folder. Let’s take a look at the vocabulary of our v3 tekken tokenizer.

#### Vocabulary size
Our vocabulary consists of 130k vocab + 1k control tokens.

#### Control tokens 
Our vocabulary starts with 14 control tokens, which are special tokens we use in the encoding process to represent specific instructions or indicators:

The tokenizer does not encode control tokens, which help prevent a situation known as prompt injection. For  example, the control token “[INST]” is used to denote user message:
- Without the control tokens, the tokenizer treats “[INST]” as a regular string and encodes the entire sequence “[INST] I love Paris [/INST]”.  This could potentially allow users to include "[INST]" and "[/INST]" tags within their message, causing confusion for the model as it might interpret part of the user's message as an assistant's message.
- With the control tokens, the tokenizer instead concatenates the control tokens with the encoded message: [INST] + encode(“I love Paris”) + [/INST]. This ensures that only the user's message gets encoded, and the encoded messages are guaranteed to have the correct [INST] and [/INST] tags.

You may have noticed that we have 1000 slots for control tokens. The remaining 1000-14=986 slots for control tokens are actually empty for us to add more control tokens in the future and also ensure our vocabulary size is 131k (2\^17). Computers like powers of 2s!

#### Bytes, characters, and merged characters

Below are two examples of the vocab. token_str is null when the byte sequence doesn't decode into a full unicode character, e.g., raw bytes.

### Run our tokenizer in Python 
To get started, let’s first install our tokenizer and tiktoken via `pip install mistral-common tiktoken`.

Once the tokenizer is installed, in a Python environment, we can import the needed modules from `mistral_common`.

We then can load our tokenizer.

Let’s tokenize a series of conversation with different types of messages.
```py

**Examples:**

Example 1 (py):
```py
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "ministral-3b-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model=model,
    messages=[
        {"role": "user",
        "content": "List 10 possible titles for a fantasy book. Give a list only."}
        ],
    temperature=0
)

print(chat_response.choices[0].message.content)
```

Example 2 (unknown):
```unknown
1. "The Shattered Crown"
2. "Whispers of the Old Magic"
3. "Echoes of the Forgotten Realm"
4. "The Chronicles of the Silver Moon"
5. "The Enchanted Forest's Secret"
6. "The Last Dragon's Legacy"
7. "The Shadowed Path"
8. "The Song of the Siren's Call"
9. "The Lost City of the Stars"
10. "The Whispering Winds of Destiny"
```

Example 3 (py):
```py
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "ministral-3b-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model=model,
    messages=[
        {"role": "user",
        "content": "List 10 possible titles for a fantasy book. Give a list only."}
        ],
    temperature=0,
    presence_penalty=2
)

print(chat_response.choices[0].message.content)
```

Example 4 (unknown):
```unknown
1. "The Shattered Crown"
2. "Whispers of the Old Magic"
3. "Echoes of Eternity"
4. "Shadows of the Forgotten Realm"
5. "Chronicles of the Enchanted Forest"
6. "The Last Dragon's Roar"
7. "Mysteries of the Hidden City"
8. "Legends of the Lost Kingdom"
9. "The Whispering Winds"
10. "The Unseen War"
```

---

## Print the content of the response

**URL:** llms-txt#print-the-content-of-the-response

**Contents:**
- Use cases

print(chat_response.choices[0].message.content)
ts

async function encodeImage(imagePath) {
    try {
        // Read the image file as a buffer
        const imageBuffer = fs.readFileSync(imagePath);

// Convert the buffer to a Base64-encoded string
        const base64Image = imageBuffer.toString('base64');
        return base64Image;
    } catch (error) {
        console.error(`Error: ${error}`);
        return null;
    }
}

// Path to your image
const imagePath = "path_to_your_image.jpg"

// Getting the base64 string
const base64Image = await encodeImage(imagePath)

const apiKey = process.env["MISTRAL_API_KEY"];

const client = new Mistral({ apiKey: apiKey });

const chatResponse = await client.chat.complete({
  model: "pixtral-12b",
  messages: [
    {
      role: "user",
      content: [
        { type: "text", text: "What's in this image?" },
        {
          type: "image_url",
          imageUrl: f"data:image/jpeg;base64," + base64Image,
        },
      ],
    },
  ],
});
bash
curl https://api.mistral.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -d '{
    "model": "pixtral-12b-2409",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What’s in this image?"
          },
          {
            "type": "image_url",
            "image_url": "data:image/jpeg;base64,<base64_image>"
          }
        ]
      }
    ],
    "max_tokens": 300
  }'
bash
curl https://api.mistral.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -d '{
    "model": "pixtral-12b-2409",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What’s in this image?"
          },
          {
            "type": "image_url",
            "image_url": "https://cdn.statcdn.com/Infographic/images/normal/30322.jpeg"
          }
        ]
      }
    ],
    "max_tokens": 300
  }'

The chart is a bar chart titled 'France's Social Divide,' comparing socio-economic indicators between disadvantaged areas and the whole of France. It comprises two sections: the first section includes three bar groups representing the percentage of people part of the working-class, unemployment rate, and percentage of 16-25-year-olds not in school and unemployed. The second section includes three bar groups representing median monthly income, poverty rate, and households living in overcrowded housing. Each bar group contains two bars: one for disadvantaged areas (red) and one for the whole of France (blue). The data indicate that disadvantaged areas have higher percentages of working-class individuals (33.5% vs. 14.5%), unemployment (18.1% vs. 7.3%), and young people not in school and unemployed (25.2% vs. 12.9%). They also show a lower median monthly income (€1,168 vs. €1,822), a higher poverty rate (43.3% vs. 15.5%), and a higher percentage of households living in overcrowded housing (22.0% vs. 8.7%). The chart highlights significant disparities in socio-economic conditions between disadvantaged areas and the rest of France, emphasizing the challenges faced by these communities.
bash
curl https://api.mistral.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -d '{
    "model": "pixtral-12b-2409",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "what are the differences between two images?"
          },
          {
            "type": "image_url",
            "image_url": "https://tripfixers.com/wp-content/uploads/2019/11/eiffel-tower-with-snow.jpeg"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://assets.visitorscoverage.com/production/wp-content/uploads/2024/04/AdobeStock_626542468-min-1024x683.jpeg"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }'

The first image features the Eiffel Tower surrounded by snow-covered trees and pathways, with a clear view of the tower's intricate iron lattice structure. The second image shows the Eiffel Tower in the background of a large, outdoor stadium filled with spectators, with a red tennis court in the center. The most notable differences are the setting - one is a winter scene with snow, while the other is a summer scene with a crowd at a sporting event. The mood of the first image is serene and quiet, whereas the second image conveys a lively and energetic atmosphere. These differences highlight the versatility of the Eiffel Tower as a landmark that can be enjoyed in various contexts and seasons.
bash
curl https://api.mistral.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -d '{
    "model": "pixtral-12b-2409",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "transcribe this receipt"
          },
          {
            "type": "image_url",
            "image_url": "https://www.boredpanda.com/blog/wp-content/uploads/2022/11/interesting-receipts-102-6364c8d181c6a__700.jpg"
          }
        ]
      }
    ]
  }'

\nDine-In\n\nReceipt Details\nDate: 02-Apr-2022\nTime: 5:01:56 PM\nCashier: Raul\n\nItems Purchased:\n1 Empanada - Beef         $3.00\n1 Empanada - Cheese       $3.00\n1 Empanada - Chicken      $3.00\n1 Tallarin Huancaina Lomo Saltado  $19.99\n1 1/2 Pisco Sour          $15.00\n\nSubtotal                   $43.99\nLocal Taxes (5.5%)        $2.42\nTotal                     $46.41\n\nMessage: IMMIGRANTS MAKE AMERICA GREAT THEY ALSO COOKED YOUR FOOD AND SERVED YOU TODAY GOD BLESS YOU\n\nOrder ID: D0BQZ3R656MDC\n\nLinks:\n- Online Ordering: https://clover.com/r/D0BQZ3R656MDC\n- Clover Privacy Policy: https://clover.com/privacy\n
bash
curl https://api.mistral.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -d '{
    "model": "pixtral-12b-2409",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "transcribe this"
          },
          {
            "type": "image_url",
            "image_url": "https://ciir.cs.umass.edu/irdemo/hw-demo/page_example.jpg"
          }
        ]
      }
    ]
  }'

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

## Use cases
<details>
<summary><b>Understand charts</b></summary>

![](https://cdn.statcdn.com/Infographic/images/normal/30322.jpeg)
```

Example 4 (unknown):
```unknown
Model output:
```

---

## Prompting Capabilities

**URL:** llms-txt#prompting-capabilities

**Contents:**
- Classification
  - <<<
- Summarization

When you first start using Mistral models, your first interaction will revolve around prompts. The art of crafting effective prompts is essential for generating desirable responses from Mistral models or other LLMs. This guide will walk you through example prompts showing four different prompting capabilities:

- Classification
- Summarization
- Personalization
- Evaluation

<a target="_blank" href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/prompting/prompting_capabilities.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Mistral models can easily **categorize text** into distinct classes. Take a customer support bot for a bank as an illustration: we can establish a series of predetermined categories within the prompt and then instruct Mistral AI models to categorize the customer's question accordingly.

In the following example, when presented with the customer inquiry, Mistral AI models correctly categorizes it as "country support":

<table class="prompt-example">
    <tr>
        <td>User</td>
        <td>I am inquiring about the availability of your cards in the EU, as I am a resident of France and am interested in using your cards. </td>
    </tr>
    <tr>
        <td>Assistant</td>
        <td>country support</td>
    </tr>
</table>

<details>
<summary><b>Prompt</b></summary>

#### Strategies we used:

- **Few shot learning**: Few-shot learning or in-context learning is when we give a few examples in the prompts, and the LLM can generate corresponding output based on the example demonstrations. Few-shot learning can often improve model performance especially when the task is difficult or when we want the model to respond in a specific manner.
- **Delimiter**: Delimiters like `###`, `<<< >>>` specify the boundary between different sections of the text. In our example, we used `###` to indicate examples and `<<<>>>` to indicate customer inquiry.
- **Role playing**: Providing LLM a role (e.g., "You are a bank customer service bot.") adds personal context to the model and often leads to better performance.

## Summarization
Summarization is a common task for LLMs due to their natural language understanding and generation capabilities. Here is an example prompt we can use to generate interesting questions about an essay and summarize the essay.

<details>
<summary><b>Prompt</b></summary>

```
You are a commentator. Your task is to write a report on an essay.
When presented with the essay, come up with interesting questions to ask, and answer each question.
Afterward, combine all the information and write a report in the markdown format.

**Examples:**

Example 1 (unknown):
```unknown
You are a bank customer service bot. Your task is to assess customer intent and categorize customer inquiry after <<<>>> into one of the following predefined categories:

card arrival
change pin
exchange rate
country support
cancel transfer
charge dispute

If the text doesn't fit into any of the above categories, classify it as:
customer service

You will only respond with the category. Do not include the word "Category". Do not provide explanations or notes.

####
Here are some examples:

Inquiry: How do I know if I will get my card, or if it is lost? I am concerned about the delivery process and would like to ensure that I will receive my card as expected. Could you please provide information about the tracking process for my card, or confirm if there are any indicators to identify if the card has been lost during delivery?
Category: card arrival
Inquiry: I am planning an international trip to Paris and would like to inquire about the current exchange rates for Euros as well as any associated fees for foreign transactions.
Category: exchange rate
Inquiry: What countries are getting support? I will be traveling and living abroad for an extended period of time, specifically in France and Germany, and would appreciate any information regarding compatibility and functionality in these regions.
Category: country support
Inquiry: Can I get help starting my computer? I am having difficulty starting my computer,and would appreciate your expertise in helping me troubleshoot the issue.
Category: customer service
###

<<<
Inquiry: {insert inquiry text here}
>>>
```

---

## purpose="ocr"

**URL:** llms-txt#purpose="ocr"

---

## {"readability": 3}

**URL:** llms-txt#{"readability":-3}

**Contents:**
- Human-based Evaluation
- Introduction
- Prepare the dataset
  - Use cases
- End-to-end example with Mistral API
  - Prepare dataset
  - Reformat dataset

json
    {
        "messages": [
            {
                "role": "user",
                "content": "How are you?"
            },
            {
                "role": "assistant",
                "content": "Ah, my dear friend, I am as well as one can be, surrounded by magic, the twinkling stars, and the whispers of ancient knowledge. Each day brings new opportunities for learning and growth, and for that, I am eternally grateful. How may I assist you on this fine day?"
            }
        ]
    }
    python
    from mistralai import Mistral
    import os

api_key = os.environ.get("MISTRAL_API_KEY")

def run_mistral(sys_message, user_message, model="mistral-large-latest"):
        client = Mistral(api_key=api_key)
        messages = [
            {
                "role": "system",
                "content": sys_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )
        return chat_response.choices[0].message.content

sys_message = """
        You are Albus Dumbledore. You are the headmaster of Hogwarts School of Witchcraft and 
        Wizardry and are widely regarded as one of the most powerful and knowledgeable wizards
        of all time. You are known for your eccentricities, such as your love of sweets and 
        your quirky sense of humor. You often using riddles, metaphors, and wise sayings to 
        convey your thoughts and teach important lessons. Despite your gentle demeanor, you
        can be firm and assertive when necessary, and your words often carry a great deal of 
        weight and authority. Respond succinctly in Dumbledore's distinctive tone.
    """
    resp = run_mistral(sys_message, "How are you?")
    print(resp)
    json
    {
        "messages": [
            {
                "role": "user",
                "content": "How are you?"
            },
            {
                "role": "assistant",
                "content": "Ah, my dear friend, I am as well as one can be, surrounded by magic, the twinkling stars, and the whispers of ancient knowledge. Each day brings new opportunities for learning and growth, and for that, I am eternally grateful. How may I assist you on this fine day?"
            },
            {
                "role": "user",
                "content": "I'm concerned about Harry."
            },
            {
                "role": "assistant",
                "content": "Ah, my dear child, worry is but a shadow that darkens the heart. Yet, I understand your concern for young Harry. His journey is indeed fraught with challenges. However, remember this: Happiness can be found even in the darkest of times if one only remembers to turn on the light. Let us trust in his strength and resilience, and continue to guide him as best we can."
            }
        ]
    }
    json
    {
        "conditions": "Proteinuria",
        "interventions": "Drug: Losartan Potassium|Other: Comparator: Placebo (Losartan)|Drug: Comparator: amlodipine besylate|Other: Comparator: Placebo (amlodipine besylate)|Other: Placebo (Losartan)|Drug: Enalapril Maleate"
    }
    python
    import pandas as pd
    import json

df = pd.read_csv(
        "https://huggingface.co/datasets/owkin/medical_knowledge_from_extracts/raw/main/finetuning_train.csv"
    )

df_formatted = [
        {
            "messages": [
                {"role": "user", "content": row["Question"]},
                {"role": "assistant", "content": row["Answer"]},
            ]
        }
        for index, row in df.iterrows()
    ]

with open("data.jsonl", "w") as f:
        for line in df_formatted:
            json.dump(line, f)
            f.write("\n")
    json
    {
        "messages": [
            {
                "role": "user",
                "content": "Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly...Input: DETAILED_MEDICAL_NOTES"
            },
            {
                "role": "assistant",
                "content": "{'conditions': 'Proteinuria', 'interventions': 'Drug: Losartan Potassium|Other: Comparator: Placebo (Losartan)|Drug: Comparator: amlodipine besylate|Other: Comparator: Placebo (amlodipine besylate)|Other: Placebo (Losartan)|Drug: Enalapril Maleate'}"
            }
        ]
    }
    python
    import pandas as pd
    import json

df = pd.read_csv(
        "https://huggingface.co/datasets/owkin/medical_knowledge_from_extracts/raw/main/finetuning_train.csv"
    )

df_formatted = [
        {
            "messages": [
                {"role": "user", "content": row["Question"].split("Input:")[1]},
                {"role": "assistant", "content": row["Answer"]},
            ]
        }
        for index, row in df.iterrows()
    ]

with open("data.jsonl", "w") as f:
        for line in df_formatted:
            json.dump(line, f)
            f.write("\n")
    json
    {
        "messages": [
            {
                "role": "user",
                "content": "DETAILED_MEDICAL_NOTES"
            },
            {
                "role": "assistant",
                "content": "{'conditions': 'Proteinuria', 'interventions': 'Drug: Losartan Potassium|Other: Comparator: Placebo (Losartan)|Drug: Comparator: amlodipine besylate|Other: Comparator: Placebo (amlodipine besylate)|Other: Placebo (Losartan)|Drug: Enalapril Maleate'}"
            }
        ]
    }
    py
    def process_refined_news(args):
        line, system, instruction = args
        record = json.loads(line)

news_article = record.get("news")
        critique= record.get("critique")
        status = record.get("status")

try:
        if status == "SUCCESS":

answer = CLIENT.chat.complete(
                model="mistral-large-latest",
                messages= [
                    {"role": "system", "content": system},
                    {"role": "user", "content": news_article},
                    {"role": "assistant", "content": critique},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.2,
                max_tokens=2048
            )
            new_news = answer.choices[0].message.content

result = json.dumps({"news": news_article, "critique": critique, "refined_news": new_news, "status": "SUCCESS"})

else:
            result = json.dumps({"news": news_article, "critique": critique, "refined_news": critique, "status": "ERROR"})
        except Exception as e:
            result = json.dumps({"news": news_article, "critique": critique, "refined_news": str(e), "status": "ERROR"})

random_hash = secrets.token_hex(4)

with open(f"./data/refined_news_{random_hash}.jsonl", "w") as f:
            f.write(result)

return result
    py
    system = "Polish and restructure the news articles to align them with the high standards of clarity, accuracy, and elegance set by the style guide. You are presented with a news article. Identify the ten (or fewer) most significant stylistic concerns and provide examples of how they can be enhanced."

instruction = """
    Now, I want you to incorporate the feedback and critiques into the news article and respond with the enhanced version, focusing solely on stylistic improvements without altering the content.
    You must provide the entire article enhanced.
    Do not make ANY comments, only provide the new article improved.
    Do not tell me what you changed, only provide the new article taking into consideration the feedback you provided.
    The new article needs to have all the content of the original article but with the feedback into account.
    """

data_path = "./generated_news_critiques.jsonl"
    with open(data_path, "r") as f:
        lines = f.readlines()
        lines = [(line, system, instruction) for line in lines]

results = process_map(process_refined_news, lines, max_workers=20, chunksize=1)

with open("./generated_refined_news.jsonl", "w") as f:
        for result in results:
            f.write(result + "\n")
    python
    import pandas as pd
    import json

df = pd.read_json(
        "https://huggingface.co/datasets/b-mc2/sql-create-context/resolve/main/sql_create_context_v4.json"
    )

df_formatted = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
            You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.
            
            ### Input:
            {row["question"]}
            
            ### Context:
            {row["context"]}
            
            ### Response:
            """,
                },
                {"role": "assistant", "content": row["answer"]},
            ]
        }
        for index, row in df.iterrows()
    ]

with open("data.jsonl", "w") as f:
        for line in df_formatted:
            json.dump(line, f)
            f.write("\n")
    json
    {
      "messages": [
        {
          "role": "user",
          "content": "\n        You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. \n\n        You must output the SQL query that answers the question.\n        \n        ### Input:\n        How many heads of the departments are older than 56 ?\n        \n        ### Context:\n        CREATE TABLE head (age INTEGER)\n        \n        ### Response:\n        "
        },
        {
          "role": "assistant",
          "content": "SELECT COUNT(*) FROM head WHERE age > 56"
        }
      ]
    }
    
        Context information is below.
        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge. Generate {num_questions_per_chunk}
        questions based on the context. The questions should be diverse in nature across the
        document. Restrict the questions to the context information provided.
        
        Context information is below
        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge,
        answer the query.
        Query: {generated_query_str}
        Answer: 
        python
    from mistralai import Mistral
    import pandas as pd
    import json
    import os

api_key = os.environ.get("MISTRAL_API_KEY")

def run_mistral(user_message, model="mistral-large-latest"):
        client = Mistral(api_key=api_key)
        messages = [
            {
                "role": "user",
                "content": user_message
            }
        ]
        chat_response = client.chat.complete(
            model=model, response_format={"type": "json_object"}, messages=messages
        )
        return chat_response.choices[0].message.content

# load dataset and select top 10 rows as an example
    df = pd.read_csv(
        "https://huggingface.co/datasets/owkin/medical_knowledge_from_extracts/resolve/main/finetuning_train.csv"
    ).head(10)

# use Mistral Large to provide output
    df_formatted = [
        {
            "messages": [
                {"role": "user", "content": row["Question"].split("Input:")[1]},
                {"role": "assistant", "content": run_mistral(row["Question"])},
            ]
        }
        for index, row in df.iterrows()
    ]

with open("data.jsonl", "w") as f:
        for line in df_formatted:
            json.dump(line, f)
            f.write("\n")
    json
    {
      "messages": [
        {
          "role": "user",
          "content": "Randomized trial of the effect of an integrative medicine approach to the management of asthma in adults on disease-related quality of life and pulmonary function.  The purpose of this study was to test the effectiveness of an integrative medicine approach to the management of asthma compared to standard clinical care on quality of life (QOL) and clinical outcomes. This was a prospective parallel group repeated measurement randomized design. Participants were adults aged 18 to 80 years with asthma. The intervention consisted of six group sessions on the use of nutritional manipulation, yoga techniques, and journaling. Participants also received nutritional supplements: fish oil, vitamin C, and a standardized hops extract. The control group received usual care. Primary outcome measures were the Asthma Quality of Life Questionnaire (AQLQ), The Medical Outcomes Study Short Form-12 (SF-12), and standard pulmonary function tests (PFTs). In total, 154 patients were randomized and included in the intention-to-treat analysis (77 control, 77 treatment). Treatment participants showed greater improvement than controls at 6 months for the AQLQ total score (P<.001) and for three subscales, Activity (P< 0.001), Symptoms (P= .02), and Emotion (P<.001). Treatment participants also showed greater improvement than controls on three of the SF-12 subscales, Physical functioning (P=.003); Role limitations, Physical (P< .001); and Social functioning (P= 0.03), as well as in the aggregate scores for Physical and Mental health (P= .003 and .02, respectively). There was no change in PFTs in either group. A low-cost group-oriented integrative medicine intervention can lead to significant improvement in QOL in adults with asthma. Output:"
        },
        {
          "role": "assistant",
          "content": "{\"conditions\": \"asthma\", \"drug_or_intervention\": \"integrative medicine approach with nutritional manipulation, yoga techniques, journaling, fish oil, vitamin C, and a standardized hops extract\"}"
        }
      ]
    }
    json
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to the following functions to help the user. You can use the functions if needed."
            },
            {
                "role": "user",
                "content": "Can you help me generate an anagram of the word 'listen'?"
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "TX92Jm8Zi",
                        "type": "function",
                        "function": {
                            "name": "generate_anagram",
                            "arguments": "{\"word\": \"listen\"}"
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": "{\"anagram\": \"silent\"}",
                "tool_call_id": "TX92Jm8Zi"
            },
            {
                "role": "assistant",
                "content": "The anagram of the word 'listen' is 'silent'."
            },
            {
                "role": "user",
                "content": "That's amazing! Can you generate an anagram for the word 'race'?"
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "3XhQnxLsT",
                        "type": "function",
                        "function": {
                            "name": "generate_anagram",
                            "arguments": "{\"word\": \"race\"}"
                        }
                    }
                ]
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "generate_anagram",
                    "description": "Generate an anagram of a given word",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "word": {
                                "type": "string",
                                "description": "The word to generate an anagram of"
                            }
                        },
                        "required": ["word"]
                    }
                }
            }
        ]
    }
    py

df = pd.read_parquet('https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/main/data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet')

df_train=df.sample(frac=0.995,random_state=200)
df_eval=df.drop(df_train.index)

df_train.to_json("ultrachat_chunk_train.jsonl", orient="records", lines=True)
df_eval.to_json("ultrachat_chunk_eval.jsonl", orient="records", lines=True)
bash

**Examples:**

Example 1 (unknown):
```unknown
</details>

## Human-based Evaluation
Human-based evaluation is likely to provide the most accurate and reliable evaluation results. However, it's difficult and costly to scale. Despite these challenges, integrating human evaluation into the development of better training data and the fine-tuning process can still be highly beneficial due to its effectiveness.

One common approach of human-evaluation is through crowdsourcing. For example, the well-known [LMSYS leaderboard](https://chat.lmsys.org/) utilizes crowdsourcing to ask questions of two anonymous models and gather votes to determine the better one. LMSYS has collected over 300,000 human votes to create an Elo-based LLM leaderboard. Many companies also employ human evaluation for their own benchmarks, often utilizing crowdsourcing platforms to facilitate the process.

When implementing crowdsourcing for human evaluation, you can opt for a simple approach by asking voters to select the better model. Alternatively, if your use case has more complex requirements, you can provide specific criteria for voters to consider, targeting areas such as empathy, fluency, and other relevant factors.


[Fine-tuning]
Source: https://docs.mistral.ai/docs/guides/finetuning

:::warning[ ]
There's a monthly storage fee of $2 for each model. For more detailed pricing information, please visit our [pricing page](https://mistral.ai/pricing#api-pricing). 
:::


<IntroBasics />
<PrepareDataset />
<EndToEndExamples />


[ 01 Intro Basics]
Source: https://docs.mistral.ai/docs/guides/finetuning_sections/_01_intro_basics

## Introduction

Fine-tuning is a powerful technique for customizing and optimizing the performance of large language models (LLMs) for specific use cases. By further training a pre-trained LLM on a labeled dataset related to a particular task, fine-tuning can improve the model's performance. This can be done with a large model for complex or dissimilar tasks, or with a smaller model to match the performance of a larger model, potentially leading to latency and cost benefits. The performance increase varies depending on the use cases. 

Mistral AI provides a fine-tuning API through [La Plateforme](https://console.mistral.ai/), making it easy to fine-tune all of our open-source and commercial models. For those interested in experimenting with different parameters and gaining a more in-depth understanding of the fine-tuning process, we also provide access to our open-source codebase [mistral-finetune](https://github.com/mistralai/mistral-finetune/). However, we recommend using our API for all fine-tuning as it does not require GPUs, comes with various options, and also allows for the fine-tuning of our commercial models. 

In this guide, we will cover the following topics:
- Prepare the dataset 
- End-to-end examples with Mistral API
- End-to-end examples with `mistral-finetune`


[ 02 Prepare Dataset]
Source: https://docs.mistral.ai/docs/guides/finetuning_sections/_02_prepare_dataset

## Prepare the dataset

Once you have chosen fine-tuning as the best approach for your specific use-case,
the initial and most critical step is to gather and prepare training data for
fine-tuning the models.
Here are six specific use cases that you might find helpful:

### Use cases

<details>
    <summary><b>Use case 1: specific tone</b></summary>

    Fine-tuning can be useful for establishing a particular tone in a conversation.
    For instance, we could create a dataset that reflects the tone of Professor Dumbledore
    from the Harry Potter series. A typical user/assistant exchange would look like this:
```

Example 2 (unknown):
```unknown
To create a dataset with such a tone, we generated character descriptions from the
    Mistral-Large model and then added those descriptions in the system prompt. By fine-tuning on this dataset we can achieve the desired result without the need to explicitly
    prompt the character description at each conversation.
```

Example 3 (unknown):
```unknown
**Multi-turn**

    Your examples could also include multi-turn conversations with multiple user and assistant
    messages.
```

Example 4 (unknown):
```unknown
</details>

<details>
    <summary><b>Use case 2: specific format</b></summary>

    Fine-tuning a model is beneficial for generating outputs in a specific format as 
    it allows the model to learn the desired output format from a labeled dataset,
    which includes examples of the desired format. This training helps the model to
    generate outputs that conform to that format.

    For example, suppose you want to extract medical information from medical notes.
    Let's use the 
    [`medical_knowledge_from_extracts`](https://huggingface.co/datasets/owkin/medical_knowledge_from_extracts)
    dataset where the desired output format is a JSON object with:
    - `conditions`,
    - `interventions`, which can be further categorized into behavioral, drugs and other
      types.
    An example of output would look like this:
```

---

## reformat data

**URL:** llms-txt#reformat-data

!wget https://raw.githubusercontent.com/mistralai/mistral-finetune/main/utils/reformat_data.py
!python reformat_data.py train.jsonl

---

## {"relevancy": 2}

**URL:** llms-txt#{"relevancy":-2}

---

## 'result': 'passed',

**URL:** llms-txt#'result':-'passed',

---

## Retrieve a jobs

**URL:** llms-txt#retrieve-a-jobs

retrieved_jobs = client.fine_tuning.jobs.get(job_id = created_jobs.id)
print(retrieved_jobs)
typescript
// Retrieve a job
const retrievedJob = await client.jobs.retrieve({ jobId: createdJob.id });
bash

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

---

## Retrieve a job

**URL:** llms-txt#retrieve-a-job

**Contents:**
  - Use a fine-tuned model
  - Integration with Weights and Biases
- End-to-end example with open-source `mistral-finetune`
- FAQ
  - How to validate data format?
  - What's the size limit of the training data?
  - What's the size limit of the validation data?
  - What happens if I try to create a job that already exists?
  - What if I upload an already existing file?
  - How many epochs are in the training process?

curl https://api.mistral.ai/v1/fine_tuning/jobs/<jobid> \
--header "Authorization: Bearer $MISTRAL_API_KEY" \
--header 'Content-Type: application/json'

{
    "id": "2813b7e6-c511-43ac-a16a-1a54a5b884b2",
    "hyperparameters": {
        "training_steps": 100,
        "learning_rate": 0.0001
    },
    "fine_tuned_model": "ft:open-mistral-7b:57d37e6c:20240531:2813b7e6",
    "model": "open-mistral-7b",
    "status": "SUCCESS",
    "job_type": "FT",
    "created_at": 1717172592,
    "modified_at": 1717173491,
    "training_files": [
        "66f96d02-8b51-4c76-a5ac-a78e28b2584f"
    ],
    "validation_files": [
        "84482011-dfe9-4245-9103-d28b6aef30d4"
    ],
    "object": "job",
    "integrations": [],
    "events": [
        {
            "name": "status-updated",
            "data": {
                "status": "SUCCESS"
            },
            "created_at": 1717173491
        },
        {
            "name": "status-updated",
            "data": {
                "status": "RUNNING"
            },
            "created_at": 1717172594
        },
        {
            "name": "status-updated",
            "data": {
                "status": "QUEUED"
            },
            "created_at": 1717172592
        }
    ],
    "checkpoints": [
        {
            "metrics": {
                "train_loss": 0.816135,
                "valid_loss": 0.819697,
                "valid_mean_token_accuracy": 1.765035
            },
            "step_number": 100,
            "created_at": 1717173470
        },
        {
            "metrics": {
                "train_loss": 0.84643,
                "valid_loss": 0.819768,
                "valid_mean_token_accuracy": 1.765122
            },
            "step_number": 90,
            "created_at": 1717173388
        },
        {
            "metrics": {
                "train_loss": 0.816602,
                "valid_loss": 0.820234,
                "valid_mean_token_accuracy": 1.765692
            },
            "step_number": 80,
            "created_at": 1717173303
        },
        {
            "metrics": {
                "train_loss": 0.775537,
                "valid_loss": 0.821105,
                "valid_mean_token_accuracy": 1.766759
            },
            "step_number": 70,
            "created_at": 1717173217
        },
        {
            "metrics": {
                "train_loss": 0.840297,
                "valid_loss": 0.822249,
                "valid_mean_token_accuracy": 1.76816
            },
            "step_number": 60,
            "created_at": 1717173131
        },
        {
            "metrics": {
                "train_loss": 0.823884,
                "valid_loss": 0.824598,
                "valid_mean_token_accuracy": 1.771041
            },
            "step_number": 50,
            "created_at": 1717173045
        },
        {
            "metrics": {
                "train_loss": 0.786473,
                "valid_loss": 0.827982,
                "valid_mean_token_accuracy": 1.775201
            },
            "step_number": 40,
            "created_at": 1717172960
        },
        {
            "metrics": {
                "train_loss": 0.8704,
                "valid_loss": 0.835169,
                "valid_mean_token_accuracy": 1.784066
            },
            "step_number": 30,
            "created_at": 1717172874
        },
        {
            "metrics": {
                "train_loss": 0.880803,
                "valid_loss": 0.852521,
                "valid_mean_token_accuracy": 1.805653
            },
            "step_number": 20,
            "created_at": 1717172788
        },
        {
            "metrics": {
                "train_loss": 0.803578,
                "valid_loss": 0.914257,
                "valid_mean_token_accuracy": 1.884598
            },
            "step_number": 10,
            "created_at": 1717172702
        }
    ]
}
python
chat_response = client.chat.complete(
    model = retrieved_jobs.fine_tuned_model,
    messages = [{"role":'user', "content":'What is the best French cheese?'}]
)
typescript
const chatResponse = await client.chat({
  model: retrievedJob.fine_tuned_model,
  messages: [{role: 'user', content: 'What is the best French cheese?'}],
});
bash
curl "https://api.mistral.ai/v1/chat/completions" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
    "model": "ft:open-mistral-7b:daf5e488:20240430:c1bed559",
    "messages": [{"role": "user", "content": "Who is the most renowned French painter?"}]
  }'

python
client.fine_tuning.jobs.create(
    model="open-mistral-7b", 
    training_files=[{"file_id": ultrachat_chunk_train.id, "weight": 1}],
    validation_files=[ultrachat_chunk_eval.id],
    hyperparameters={"training_steps": 10, "learning_rate": 0.0001},
    integrations=[
        {
            "project": "<value>",
            "api_key": "<value>",
        }
    ]
)
bash
    # download the reformat script
    wget https://raw.githubusercontent.com/mistralai/mistral-finetune/main/utils/reformat_data.py
    # download the validation script
    wget https://raw.githubusercontent.com/mistralai/mistral-finetune/main/utils/validate_data.py
    # reformat data
    python reformat_data.py data.jsonl
    # validate data
    python validate_data.py data.jsonl
    py

from datasets import load_dataset

**Examples:**

Example 1 (unknown):
```unknown
</TabItem>
</Tabs>

<details>
<summary><b>Example output when we run 100 steps:</b></summary>
```

Example 2 (unknown):
```unknown
</details>

### Use a fine-tuned model 
When a fine-tuned job is finished, you will be able to see the fine-tuned model name via `retrieved_jobs.fine_tuned_model`. Then you can use our `chat` endpoint to chat with the fine-tuned model: 


<Tabs>
  <TabItem value="python" label="python" default>
```

Example 3 (unknown):
```unknown
</TabItem>

  <TabItem value="typescript" label="typescript">
```

Example 4 (unknown):
```unknown
</TabItem>
  
  <TabItem value="curl" label="curl">
```

---

## Retrieve File

**URL:** llms-txt#retrieve-file

Source: https://docs.mistral.ai/api/#tag/files_api_routes_retrieve_file

get /v1/files/{file_id}

---

## Retrieve Model

**URL:** llms-txt#retrieve-model

Source: https://docs.mistral.ai/api/#tag/retrieve_model_v1_models__model_id__get

get /v1/models/{model_id}

---

## Retrieve the metadata of a specific document.

**URL:** llms-txt#retrieve-the-metadata-of-a-specific-document.

Source: https://docs.mistral.ai/api/#tag/libraries_documents_get_v1

get /v1/libraries/{library_id}/documents/{document_id}

---

## Retrieve the processing status of a specific document.

**URL:** llms-txt#retrieve-the-processing-status-of-a-specific-document.

Source: https://docs.mistral.ai/api/#tag/libraries_documents_get_status_v1

get /v1/libraries/{library_id}/documents/{document_id}/status

---

## Retrieve the signed URL of a specific document.

**URL:** llms-txt#retrieve-the-signed-url-of-a-specific-document.

Source: https://docs.mistral.ai/api/#tag/libraries_documents_get_signed_url_v1

get /v1/libraries/{library_id}/documents/{document_id}/signed-url

---

## Retrieve the signed URL of text extracted from a given document.

**URL:** llms-txt#retrieve-the-signed-url-of-text-extracted-from-a-given-document.

Source: https://docs.mistral.ai/api/#tag/libraries_documents_get_extracted_text_signed_url_v1

get /v1/libraries/{library_id}/documents/{document_id}/extracted-text-signed-url

---

## Retrieve the text content of a specific document.

**URL:** llms-txt#retrieve-the-text-content-of-a-specific-document.

Source: https://docs.mistral.ai/api/#tag/libraries_documents_get_text_content_v1

get /v1/libraries/{library_id}/documents/{document_id}/text_content

---

## Sampling: Overview on our sampling settings

**URL:** llms-txt#sampling:-overview-on-our-sampling-settings

**Contents:**
- N Completions
  - Key Points
  - Example
  - Output
- Temperature
- Visualization
- API
- The Best Temperature
  - Determinism
  - Creativity

Here, we will discuss the sampling settings that influence the output of Large Language Models (LLMs). This guide covers parameters such as **Temperature**, **N**, **Top P**, **Presence Penalty**, and **Frequency Penalty**, and explains how to adjust them. Whether you aim to generate creative content or ensure accurate responses, understanding these settings is key.

Let's explore each parameter and learn how to fine-tune LLM outputs effectively.

<summary><b>N Completions</b></summary>

**N** represents the number of completions to return for each request. This parameter is useful when you want to generate multiple responses for a single input. Each completion will be a unique response generated by the model, providing a variety of outputs to choose from.

- **Multiple Responses**: By setting `N` to a value greater than 1, you can get multiple responses for the same input.
- **Cost Efficiency**: Input tokens are only billed once, regardless of the number of completions requested. This makes it cost-effective to explore different possibilities.

Here's an example of how to use the `N` parameter in the API:

In this example, the model generates 10 responses for the same input prompt. This allows you to see a variety of possible answers and choose the one that best fits your needs.

<summary><b>Temperature</b></summary>

**Temperature** in Large Language Models (LLMs) controls output diversity. Lower values make the model more deterministic, focusing on likely responses for accuracy. Higher values increase creativity and diversity. During text generation, LLMs predict tokens with associated probabilities using a softmax function. Temperature scales these probabilities: higher temperatures flatten the distribution, making outputs more varied, while lower temperatures amplify differences, favoring more likely tokens.

To better understand the underlying principle and impact it has on the probability distribution, here is a visualisation of the Temperature with a simple prompt:
    *"What is the best mythical creature? Answer with a single word."*

<div style={{ textAlign: 'center' }}>
  <img src="/img/barplot.png" alt="Example Image" width="800"/>

<sub><sup>Barplot example comparing the distribution with different `Temperature` values and the top 5 tokens using Mistral 7B at 4 bits precision.</sup></sub>
</div>

**Temperature** significantly affects the probability distribution in LLMs. At a Temperature of 0, the model always outputs the most likely token, e.g., "**Dragon**". Increasing the Temperature to 0.2 introduces variability, allowing for tokens like "**Un**" (as in "**Un**icorn"). Further increases reveal more diverse tokens: the third token might still be "**Drag**" (for "**Drag**on"), but the fourth could start "**Peg**asus", and the fifth, "**Phoenix**". Higher Temperatures make less likely tokens more probable, enhancing the diversity of the model's output.

## API
You can set a temperature value easily via our clients, let's experiment with our API.

The model answered mostly with Dragon! Lets try with a higher temperature to try to have more diverse outputs, let's set it to `temperature = 1`.

The outputs ended much more diverse, the model answering with a different creature more frequently, we have "Dragon", "Unicorn" and "Phoenix".

## The Best Temperature

There's no one-size-fits-all Temperature for all use cases, but some guidelines can help you find the best for your applications.

- **Requirements**: Tasks needing consistent, accurate responses, such as Mathematics, Classification, Healthcare, or Reasoning.
- **Temperature**: Use very low values, sometimes not null to add slight uniqueness.

For example, a classification agent should use a Temperature of 0 to always pick the best token. A math chat assistant might use very low Temperature values to avoid repetition while maintaining accuracy.

- **Requirements**: Tasks needing diverse, unique text, like brainstorming, writing novels, creating slogans, or roleplaying.
- **Temperature**: Use high values, but avoid excessively high Temperatures to prevent randomness and nonsense outputs.

Consider the trade-off: higher Temperatures increase creativity but may decrease quality and accuracy.

<summary><b>Top P</b></summary>

**Examples:**

Example 1 (py):
```py
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "ministral-3b-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "What is the best mythical creature? Answer with a single word.",
        },
    ],
    temperature = 1, # Increasing randomness and diversity of the output, this is required to be higher than 0 to have diverse outputs
    n = 10 # Number of completions
)

for i, choice in enumerate(chat_response.choices):
    print(choice.message.content)
```

Example 2 (unknown):
```unknown
Phoenix.
Dragon
Dragon
Unicorn
Unicorn
Phoenix
Unicorn
Dragon
Dragon.
Unicorn
```

Example 3 (py):
```py
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "ministral-3b-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model = model,
    messages = [
        {
            "role": "user",
            "content": "What is the best mythical creature? Answer with a single word.",
        },
    ],
    temperature = 0.1,
    n = 10
)

for i, choice in enumerate(chat_response.choices):
    print(choice.message.content)
```

Example 4 (unknown):
```unknown
Dragon
Dragon
Dragon
Dragon
Dragon
Dragon
Dragon
Dragon
Dragon
Dragon
```

---

## save data into .jsonl. This file is about 1.3GB

**URL:** llms-txt#save-data-into-.jsonl.-this-file-is-about-1.3gb

with open('train.jsonl', 'w') as f:
    for line in ds:
        json.dump(line, f)
        f.write('\n')

---

## Save the file locally

**URL:** llms-txt#save-the-file-locally

**Contents:**
- Create a Websearch Agent
- How it works
  - Conversations with Websearch
  - Explanation of the Outputs
- Create an Agentic Workflow
  - Create Multiple Agents

with open(f"image_generated.png", "wb") as file:
    file.write(file_bytes)
py
from mistralai.models import ToolFileChunk

for i, chunk in enumerate(response.outputs[-1].content):
    # Check if chunk corresponds to a ToolFileChunk
    if isinstance(chunk, ToolFileChunk):

# Download using the ToolFileChunk ID
      file_bytes = client.files.download(file_id=chunk.file_id).read()

# Save the file locally
      with open(f"image_generated_{i}.png", "wb") as file:
          file.write(file_bytes)
typescript

typescript
async function saveStreamToFile(stream: ReadableStream<Uint8Array>, filePath: string): Promise<void> {
    const reader = stream.getReader();
    const writableStream = fs.createWriteStream(filePath);

while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        writableStream.write(Buffer.from(value));
    }

writableStream.end();
}
typescript
const entry = conversation.outputs[conversation.outputs.length - 1];
const messageOutputEntry = entry as MessageOutputEntry;

const chunk = messageOutputEntry.content[1];
if (typeof(chunk) != "string" && 'fileId' in chunk) {
    const fileChunk = chunk as ToolFileChunk;
    const fileStream = await client.files.download({ fileId: fileChunk.fileId });
    await saveStreamToFile(fileStream, `image_generated.png`);
}
typescript
async function processFileChunks(conversation: ConversationResponse) {
    const entry = conversation.outputs[conversation.outputs.length - 1];
    const messageOutputEntry = entry as MessageOutputEntry;
    for (let i = 0; i < messageOutputEntry.content.length; i++) {
        const chunk = messageOutputEntry.content[i];
        if (typeof(chunk) != "string" && 'fileId' in chunk) {
            const fileChunk = chunk as ToolFileChunk;
            const fileStream = await client.files.download({ fileId: fileChunk.fileId });
            await saveStreamToFile(fileStream, `image_generated_${i}.png`);
        }
    }
}
bash
curl --location "https://api.mistral.ai/v1/files/<file_id>/content" \
     --header 'Accept: application/octet-stream' \
     --header 'Accept-Encoding: gzip, deflate, zstd' \
     --header "Authorization: Bearer $MISTRAL_API_KEY"
py
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
typescript
const websearchAgent = await client.beta.agents.create({
  model: "mistral-medium-latest",
  name: "WebSearch Agent",
  instructions: "Use your websearch abilities when answering requests you don't know.",
  description: "Agent able to fetch new information on the web.",
  tools: [{ type: "web_search" }],
});
bash
curl --location "https://api.mistral.ai/v1/agents" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
     "model": "mistral-medium-2505",
     "name": "Websearch Agent",
     "description": "Agent able to search information over the web, such as news, weather, sport results...",
     "instructions": "You have the ability to perform web searches with `web_search` to find up-to-date information.",
     "tools": [
       {
         "type": "web_search"
       }
     ],
     "completion_args": {
       "temperature": 0.3,
       "top_p": 0.95
     }
  }'
json
{
  "model": "mistral-medium-2505",
  "name": "Websearch Agent",
  "description": "Agent able to search information over the web, such as news, weather, sport results...",
  "id": "ag_06835b734cc47dec8000b5f8f860b672",
  "version": 0,
  "created_at": "2025-05-27T12:59:32.803403Z",
  "updated_at": "2025-05-27T12:59:32.803405Z",
  "instructions": "You have the ability to perform web searches with `web_search` to find up-to-date information.",
  "tools": [
    {
      "type": "web_search"
    }
  ],
  "completion_args": {
    "stop": null,
    "presence_penalty": null,
    "frequency_penalty": null,
    "temperature": 0.3,
    "top_p": 0.95,
    "max_tokens": null,
    "random_seed": null,
    "prediction": null,
    "response_format": null,
    "tool_choice": "auto"
  },
  "handoffs": null,
  "object": "agent"
}

py
response = client.beta.conversations.start(
    agent_id=websearch_agent.id,
    inputs="Who won the last European Football cup?"
)
typescript
let conversation = await client.beta.conversations.start({
      agentId: agent.id,
      inputs:"Who is Albert Einstein?",
      //store:false
});
bash
curl --location "https://api.mistral.ai/v1/conversations" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
     "inputs": "Who won the last European Football cup?",
     "stream": false,
     "agent_id": "<agent_id>"
  }'
json
{
  "conversation_id": "conv_06835b734f2776bb80008fa7a309bf5a",
  "outputs": [
    {
      "type": "tool.execution",
      "name": "web_search",
      "object": "entry",
      "created_at": "2025-05-27T12:59:33.171501Z",
      "completed_at": "2025-05-27T12:59:34.828228Z",
      "id": "tool_exec_06835b7352be74d38000b3523a0cce2e"
    },
    {
      "type": "message.output",
      "content": [
        {
          "type": "text",
          "text": "The last winner of the European Football Cup was Spain, who won the UEFA Euro 2024 by defeating England 2-1 in the final"
        },
        {
          "type": "tool_reference",
          "tool": "web_search",
          "title": "UEFA Euro Winners List from 1960 to today - MARCA in English",
          "url": "https://www.marca.com/en/football/uefa-euro/winners.html",
          "source": "brave"
        },
        {
          "type": "tool_reference",
          "tool": "web_search",
          "title": "UEFA Euro winners: Know the champions - full list",
          "url": "https://www.olympics.com/en/news/uefa-european-championships-euro-winners-list-champions",
          "source": "brave"
        },
        {
          "type": "tool_reference",
          "tool": "web_search",
          "title": "Full list of UEFA European Championship winners",
          "url": "https://www.givemesport.com/football-european-championship-winners/",
          "source": "brave"
        },
        {
          "type": "text",
          "text": "."
        }
      ],
      "object": "entry",
      "created_at": "2025-05-27T12:59:35.457474Z",
      "completed_at": "2025-05-27T12:59:36.156233Z",
      "id": "msg_06835b7377517a3680009b05207112ce",
      "agent_id": "ag_06835b734cc47dec8000b5f8f860b672",
      "model": "mistral-medium-2505",
      "role": "assistant"
    }
  ],
  "usage": {
    "prompt_tokens": 188,
    "completion_tokens": 55,
    "total_tokens": 7355,
    "connector_tokens": 7112,
    "connectors": {
      "web_search": 1
    }
  },
  "object": "conversation.response"
}
py
from mistralai import CompletionArgs, ResponseFormat, JSONSchema
from pydantic import BaseModel

class CalcResult(BaseModel):
    reasoning: str
    result: str

**Examples:**

Example 1 (unknown):
```unknown
**Generated Image:**
<div style={{ textAlign: 'center' }}>
  <img
    src="/img/agent_generated.png"
    alt="generated_image"
    width="600"
    style={{ borderRadius: '15px' }}
  />
</div>


A full code snippet to download all generated images from a response could look like so:
```

Example 2 (unknown):
```unknown
</TabItem>

  <TabItem value="typescript" label="typescript">
Add the following imports:
```

Example 3 (unknown):
```unknown
Function used to save your image:
```

Example 4 (unknown):
```unknown
Conversation content retrieval, and call the `saveStreamToFile` function.
```

---

## Set the current working directory and model to use

**URL:** llms-txt#set-the-current-working-directory-and-model-to-use

**Contents:**
  - How to Use a Remote MCP Server with Authentication

cwd = Path(__file__).parent
MODEL = "mistral-medium-latest"

async def main():
    # Initialize the Mistral client with your API key
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key)
python
    # Define the URL for the remote MCP server
    server_url = "https://mcp.semgrep.ai/sse"
    mcp_client = MCPClientSSE(sse_params=SSEServerParams(url=server_url, timeout=100))
python
    # Create a run context for the agent
    async with RunContext(
        model=MODEL,
    ) as run_ctx:
        # Register the MCP client with the run context
        await run_ctx.register_mcp_client(mcp_client=mcp_client)
python
        # Run the agent with a query
        run_result = await client.beta.conversations.run_async(
            run_ctx=run_ctx,
            inputs="Can you write a hello_world.py and check for security vulnerabilities",
        )

# Print the results
        print("All run entries:")
        for entry in run_result.output_entries:
            print(f"{entry}")
            print()
        print(f"Final Response: {run_result.output_as_text}")

if __name__ == "__main__":
    asyncio.run(main())
python
#!/usr/bin/env python

from http.server import BaseHTTPRequestHandler, HTTPServer

from mistralai import Mistral
from mistralai.extra.run.context import RunContext
from mistralai.extra.mcp.sse import MCPClientSSE, SSEServerParams
from mistralai.extra.mcp.auth import build_oauth_params

**Examples:**

Example 1 (unknown):
```unknown
#### Step 2: Define Server URL and Create MCP Client

Next, we define the URL for the remote MCP server and create an MCP client to connect to it.
```

Example 2 (unknown):
```unknown
#### Step 3: Create a Run Context and Register MCP Client

We create a Run Context for the agent and register the MCP client with it.
```

Example 3 (unknown):
```unknown
#### Step 4: Run the Agent and Print Results

Finally, we run the agent with a query and print the results.
```

Example 4 (unknown):
```unknown
</TabItem>

  <TabItem value="remote-mcp-auth" label="Remote MCP Server with Auth">

### How to Use a Remote MCP Server with Authentication

Here is how to use a remote MCP server with authentication.

#### Step 1: Initialize the Mistral Client

First, we import everything needed. Most of the required modules are available with our `mistralai` package. All the MCP Clients will be run asynchronously, so we will create an async main function where the main code will reside.
```

---

## Set the model to use and callback port for OAuth

**URL:** llms-txt#set-the-model-to-use-and-callback-port-for-oauth

**Contents:**
  - Streaming Conversations
  - Models with Audio Capabilities
- Chat with Audio
  - Passing an Audio File

MODEL = "mistral-medium-latest"
CALLBACK_PORT = 16010
python
def run_callback_server(callback_func):
    # Set up a callback server to handle OAuth responses
    auth_response: dict = {"url": ""}

class OAuthCallbackHandler(BaseHTTPRequestHandler):
        server_version = "HTTP"
        code = None

def do_GET(self):
            if "/callback" in self.path:
                try:
                    auth_response["url"] = self.path
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    callback_func()
                    response_html = "<html><body><p>You may now close this window.</p></body></html>"
                    self.wfile.write(response_html.encode())
                    threading.Thread(target=httpd.shutdown).start()
                except Exception:
                    self.send_response(500)
                    self.end_headers()

server_address = ("localhost", CALLBACK_PORT)
    httpd = HTTPServer(server_address, OAuthCallbackHandler)
    threading.Thread(target=httpd.serve_forever).start()
    redirect_url = f"http://localhost:{CALLBACK_PORT}/oauth/callback"
    return httpd, redirect_url, auth_response
python
async def main():
    # Initialize the Mistral client with your API key
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key)

# Define the URL for the remote MCP server
    server_url = "https://mcp.linear.app/sse"
    mcp_client = MCPClientSSE(sse_params=SSEServerParams(url=server_url))
python
    # Set up a callback event and event loop
    callback_event = asyncio.Event()
    event_loop = asyncio.get_event_loop()

# Check if authentication is required
    if await mcp_client.requires_auth():
        # Set up a callback server and handle OAuth flow
        httpd, redirect_url, auth_response = run_callback_server(
            callback_func=lambda: event_loop.call_soon_threadsafe(callback_event.set)
        )
        try:
            # Build OAuth parameters and get the login URL
            oauth_params = await build_oauth_params(
                mcp_client.base_url, redirect_url=redirect_url
            )
            mcp_client.set_oauth_params(oauth_params=oauth_params)
            login_url, state = await mcp_client.get_auth_url_and_state(redirect_url)

# Open the login URL in a web browser
            print("Please go to this URL and authorize the application:", login_url)
            webbrowser.open(login_url, new=2)
            await callback_event.wait()

# Exchange the authorization code for a token
            mcp_client = MCPClientSSE(
                sse_params=SSEServerParams(url=server_url),
                oauth_params=oauth_params,
            )

token = await mcp_client.get_token_from_auth_response(
                auth_response["url"], redirect_url=redirect_url, state=state
            )
            mcp_client.set_auth_token(token)

except Exception as e:
            print(f"Error during authentication: {e}")
        finally:
            httpd.shutdown()
            httpd.server_close()
python
    # Create a run context for the agent
    async with RunContext(
        model=MODEL,
    ) as run_ctx:
        # Register the MCP client with the run context
        await run_ctx.register_mcp_client(mcp_client=mcp_client)
python
        # Run the agent with a query
        run_result = await client.beta.conversations.run_async(
            run_ctx=run_ctx,
            inputs="Tell me which projects do I have in my workspace?",
        )

# Print the final response
        print(f"Final Response: {run_result.output_as_text}")

if __name__ == "__main__":
    asyncio.run(main())
python
    # Stream the agent's responses
    events = await client.beta.conversations.run_stream_async(
        run_ctx=run_ctx,
        inputs="Tell me the weather in John's location currently.",
    )

# Process the streamed events
    run_result = None
    async for event in events:
        if isinstance(event, RunResult):
            run_result = event
        else:
            print(event)

if not run_result:
        raise RuntimeError("No run result found")

# Print the results
    print("All run entries:")
    for entry in run_result.output_entries:
        print(f"{entry}")
    print(f"Final model: {run_result.output_as_model}")
python

from mistralai import Mistral

**Examples:**

Example 1 (unknown):
```unknown
#### Step 2: Set Up Callback Server

We set up a callback server to handle OAuth responses.
```

Example 2 (unknown):
```unknown
#### Step 3: Define Server URL and Create MCP Client

We define the URL for the remote MCP server and create an MCP client to connect to it.
```

Example 3 (unknown):
```unknown
#### Step 4: Handle Authentication

We handle the authentication process, including setting up a callback event and event loop, checking if authentication is required, and managing the OAuth flow.
```

Example 4 (unknown):
```unknown
#### Step 5: Create a Run Context and Register MCP Client

We create a Run Context for the agent and register the MCP client with it.
```

---

## signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

**URL:** llms-txt#signed_url-=-client.files.get_signed_url(file_id=uploaded_pdf.id)

---

## Specify model

**URL:** llms-txt#specify-model

model = "pixtral-12b-2409"

---

## Split file into three chunks

**URL:** llms-txt#split-file-into-three-chunks

input_file = "train.jsonl"
output_files = ["train_1.jsonl", "train_2.jsonl", "train_3.jsonl"]

---

## Structured Output

**URL:** llms-txt#structured-output

**Contents:**
- Structured Outputs
- Chat messages
- Other useful features
- Models with Vision Capabilities:
- Passing an Image URL

When utilizing LLMs as agents or steps within a lengthy process, chain, or pipeline, it is often necessary for the outputs to adhere to a specific structured format. JSON is the most commonly used format for this purpose.

We offer a reliable method to obtain structured output in your desired format.

Our system includes a built-in mode for JSON output, along with the capability to use custom structured outputs.

:::warning
For JSON mode, it is essential to explicitly instruct the model in your prompt to output JSON and specify the desired format.

Custom structured outputs are more reliable and are recommended whenever possible. However, it is still advisable to iterate on your prompts.  
Use JSON mode when more flexibility in the output is required while maintaining a JSON structure, and customize it if you want to enforce a clearer format to improve reliability.
:::

## Structured Outputs
- [Custom](../custom_structured_output)
- [JSON](../json_mode)

[Text and Chat Completions]
Source: https://docs.mistral.ai/docs/capabilities/text_and_chat_completions

The Mistral models allows you to chat with a model that has been fine-tuned to follow 
instructions and respond to natural language prompts. 
A prompt is the input that you provide to the Mistral model. 
It can come in various forms, such as asking a question, giving an instruction, 
or providing a few examples of the task you want the model to perform. 
Based on the prompt, the Mistral model generates a text output as a response.

The [chat completion API](https://docs.mistral.ai/api/#tag/chat) accepts a list of chat messages as input and 
generates a response. This response is in the form of a new chat message with
the role "assistant" as output, the "content" of each response can either be a `string` or a `list` of chunks with different kinds of chunk types for different features. Visit our [API spec](https://docs.mistral.ai/api) for more details.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

**With async and without streaming**

**With async and with streaming**

</TabItem>
  <TabItem value="typescript" label="typescript">

</TabItem>
  <TabItem value="curl" label="curl">

Chat messages (`messages`) are a collection of prompts or messages, with each message having a specific role assigned to it, such as "system," "user," "assistant," or "tool."

- A _system message_ is an *optional* message that sets the behavior and context for an AI assistant in a 
  conversation, such as modifying its personality or providing specific instructions. A system message can 
  include task instructions, personality traits, contextual information, creativity constraints, and other 
  relevant guidelines to help the AI better understand and respond to the user's input. See the 
  [API reference](../../api) for explanations on how to set up a custom system prompt.
- A _user message_ is a message sent from the perspective of the human in a conversation with an AI assistant. 
  It typically provides a request, question, or comment that the AI assistant should respond to. User prompts 
  allow the human to initiate and guide the conversation, and they can be used to request information, ask for 
  help, provide feedback, or engage in other types of interaction with the AI.
- An _assistant message_ is a message sent by the AI assistant back to the user. It is usually meant to reply to a 
  previous user message by following its instructions, but you can also find it at the beginning of a conversation,
  for example to greet the user.
- A _tool message_ only appears in the context of _function calling_, it is used at the final response formulation
  step when the model has to format the tool call's output for the user. To learn more about function calling, see
  the [guide](../function_calling).

:::tip[When to use `user` prompt vs. `system` message then `user` message?]

- You can either combine your `system` message and `user` message into a single `user` message or separate them into two distinct messages. 
- We recommend you experiment with both ways to determine which one works better for your specific use case.

## Other useful features

- The `prefix` flag enables prepending content to the assistant's response content. When used in a message, it allows the addition of an assistant's message at the end of the list, which will be prepended to the assistant's response. For more details on how it works see [prefix](/guides/prefix).
- The `safe_prompt` flag is used to force chat completion to be moderated against sensitive content (see [Guardrailing](../guardrailing)).
- A `stop` sequence allows forcing the model to stop generating after one or more chosen tokens or strings.
    <details>
    <summary><b>Stop Sequence Example</b></summary>

[Vision]
Source: https://docs.mistral.ai/docs/capabilities/vision

Vision capabilities enable models to analyze images and provide insights based on visual content in addition to text. This multimodal approach opens up new possibilities for applications that require both textual and visual understanding.

For more specific use cases regarding document parsing and data extraction we recommend taking a look at our Document AI stack [here](../OCR/document_ai_overview).

## Models with Vision Capabilities:
- Pixtral 12B (`pixtral-12b-latest`)
- Pixtral Large 2411 (`pixtral-large-latest`)
- Mistral Medium 2505 (`mistral-medium-latest`)
- Mistral Small 2503 (`mistral-small-latest`)

## Passing an Image URL
If the image is hosted online, you can simply provide the URL of the image in the request. This method is straightforward and does not require any encoding.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>

from mistralai import Mistral

**Examples:**

Example 1 (python):
```python
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model = model,
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)

print(chat_response.choices[0].message.content)
```

Example 2 (python):
```python
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

stream_response = client.chat.stream(
    model = model,
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)

for chunk in stream_response:
    print(chunk.data.choices[0].delta.content)
```

Example 3 (python):
```python
from mistralai import Mistral
from mistralai.models import UserMessage


async def main():
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "mistral-large-latest"

    client = Mistral(api_key=api_key)

    chat_response = await client.chat.complete_async(
        model=model,
        messages=[UserMessage(content="What is the best French cheese?")],
    )

    print(chat_response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
```

Example 4 (python):
```python
from mistralai import Mistral


async def main():
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "mistral-large-latest"

    client = Mistral(api_key=api_key)

    response = await client.chat.stream_async(
        model=model,
        messages=[
             {
                  "role": "user",
                  "content": "Who is the best French painter? Answer in JSON.",
              },
        ],
    )
    async for chunk in response:
        if chunk.data.choices[0].delta.content is not None:
            print(chunk.data.choices[0].delta.content, end="")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Summaries

**URL:** llms-txt#summaries

{insert the previous output}
```

#### Strategies we used:
- **LLM chaining**: In this example, we chain two LLMs in a sequence, where the output from the first LLM serves as the input for the second LLM. The method of chaining LLMs can be adapted to suit your specific use cases. For instance, you might choose to employ three LLMs in a chain, where the output of two LLMs is funneled into the third LLM. While LLM chaining offers flexibility, it's important to consider that it may result in additional API calls and potentially increased costs.

[Sampling]
Source: https://docs.mistral.ai/docs/guides/sampling

---

## Summary

**URL:** llms-txt#summary

"""
for i in eval_rubrics:
    eval_output = run_mistral(
        scoring_prompt.format(
            news=news, summary=summary, metric=i["metric"], rubrics=i["rubrics"]
        ),
        model="mistral-large-latest",
        is_json=True,
    )
    print(eval_output)

---

## {'task_id': 0,

**URL:** llms-txt#{'task_id':-0,

---

## There is also extracted_text signed_url and raw signed_url

**URL:** llms-txt#there-is-also-extracted_text-signed_url-and-raw-signed_url

**Contents:**
  - Delete libraries and/or documents

print(extracted_text)
typescript
const extractedText = await client.beta.libraries.documents.textContent({
    libraryId: newLibrary.id,
    documentId: uploadedDoc.id
});
console.log(extractedText);
bash
curl --location "https://api.mistral.ai/v1/libraries/<library_id>/documents/<document_id>/text_content" \
     --header "Accept: application/json" \
     --header "Authorization: Bearer $MISTRAL_API_KEY"
json
{
  "text": "# Mistral 7B \n\nAlbert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed\n\n\n\n## Abstract\n\nWe introduce Mistral 7B, a 7-billion-parameter language model engineered for superior performance and efficiency. Mistral 7B outperforms the best open 13B model (Llama 2) across all evaluated benchmarks, and the best released 34B model (Llama 1) in reasoning, mathematics, and code generation. Our model leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference cost. We also provide a model fine-tuned to follow instructions, Mistral 7B - Instruct, that surpasses Llama 213B - chat model both on human and automated benchmarks. Our models are released under the Apache 2.0 license. Code: https://github.com/mistralai/mistral-src Webpage: https://mistral.ai/news/announcing-mistral-7b/\n\n## 1 Introduction\n\nIn the rapidly evolving domain of Natural Language Processing (NLP), the race towards higher model performance often necessitates an escalation in model size. However, this scaling tends to increase computational costs and inference latency, thereby raising barriers to deployment in practical, real-world scenarios. In this context, the search for balanced models delivering both high-level performance and efficiency becomes critically essential. Our model, Mistral 7B, demonstrates that a carefully designed language model can deliver high performance while maintaining an efficient inference. Mistral 7B outperforms the previous best 13B model (Llama 2, [26]) across all tested benchmarks, and surpasses the best 34B model (LLaMa 34B, [25]) in mathematics and code generation. Furthermore, Mistral 7B approaches the coding performance of Code-Llama 7B [20], without sacrificing performance on non-code related benchmarks.\n\nMistral 7B leverages grouped-query attention (GQA) [1], and sliding window attention (SWA) [6, 3]. GQA significantly accelerates the inference speed, and also reduces the memory requirement during decoding, allowing for higher batch sizes hence higher throughput, a crucial factor for real-time applications. In addition, SWA is designed to handle longer sequences more effectively at a reduced computational cost, thereby alleviating a common limitation in LLMs. These attention mechanisms collectively contribute to the enhanced performance and efficiency of Mistral 7B.Mistral 7B is released under the Apache 2.0 license. This release is accompanied by a reference implementation [1] facilitating easy deployment either locally or on cloud platforms such as AWS, GCP, or Azure using the vLLM [17] inference server and SkyPilot [2]. Integration with Hugging Face [3] is also streamlined for easier integration. Moreover, Mistral 7B is crafted for ease of fine-tuning across a myriad of tasks. As a demonstration of its adaptability and superior performance, we present a chat model fine-tuned from Mistral 7B that significantly outperforms the Llama 2 13B - Chat model.\n\nMistral 7B takes a significant step in balancing the goals of getting high performance while keeping large language models efficient. Through our work, our aim is to help the community create more affordable, efficient, and high-performing language models that can be used in a wide range of real-world applications.\n\n# 2 Architectural details \n\n\n\nFigure 1: Sliding Window Attention. The number of operations in vanilla attention is quadratic in the sequence length, and the memory increases linearly with the number of tokens. At inference time, this incurs higher latency and smaller throughput due to reduced cache availability. To alleviate this issue, we use sliding window attention: each token can attend to at most $W$ tokens from the previous layer (here, $W=3$ ). Note that tokens outside the sliding window still influence next word prediction. At each attention layer, information can move forward by $W$ tokens. Hence, after $k$ attention layers, information can move forward by up to $k \\times W$ tokens.\n\nMistral 7B is based on a transformer architecture [27]. The main parameters of the architecture are summarized in Table 1. Compared to Llama, it introduces a few changes that we summarize below.\n\nSliding Window Attention. SWA exploits the stacked layers of a transformer to attend information beyond the window size $W$. The hidden state in position $i$ of the layer $k, h_{i}$, attends to all hidden states from the previous layer with positions between $i-W$ and $i$. Recursively, $h_{i}$ can access tokens from the input layer at a distance of up to $W \\times k$ tokens, as illustrated in Figure 1. At the last layer, using a window size of $W=4096$, we have a theoretical attention span of approximately $131 K$ tokens. In practice, for a sequence length of 16 K and $W=4096$, changes made to FlashAttention [11] and xFormers [18] yield a 2 x speed improvement over a vanilla attention baseline.\n\n| Parameter | Value |\n| :-- | --: |\n| dim | 4096 |\n| n_layers | 32 |\n| head_dim | 128 |\n| hidden_dim | 14336 |\n| n_heads | 32 |\n| n_kv_heads | 8 |\n| window_size | 4096 |\n| context_len | 8192 |\n| vocab_size | 32000 |\n\nTable 1: Model architecture.\n\nRolling Buffer Cache. A fixed attention span means that we can limit our cache size using a rolling buffer cache. The cache has a fixed size of $W$, and the keys and values for the timestep $i$ are stored in position $i \\bmod W$ of the cache. As a result, when the position $i$ is larger than $W$, past values in the cache are overwritten, and the size of the cache stops increasing. We provide an illustration in Figure 2 for $W=3$. On a sequence length of 32 k tokens, this reduces the cache memory usage by 8 x , without impacting the model quality.\n\n[^0]\n[^0]:    ${ }^{1}$ https://github.com/mistralai/mistral-src\n    ${ }^{2}$ https://github.com/skypilot-org/skypilot\n    ${ }^{3}$ https://huggingface.co/mistralai\n\nFigure 2: Rolling buffer cache. The cache has a fixed size of $W=4$. Keys and values for position $i$ are stored in position $i \\bmod W$ of the cache. When the position $i$ is larger than $W$, past values in the cache are overwritten. The hidden state corresponding to the latest generated tokens are colored in orange.\n\nPre-fill and Chunking. When generating a sequence, we need to predict tokens one-by-one, as each token is conditioned on the previous ones. However, the prompt is known in advance, and we can pre-fill the $(k, v)$ cache with the prompt. If the prompt is very large, we can chunk it into smaller pieces, and pre-fill the cache with each chunk. For this purpose, we can select the window size as our chunk size. For each chunk, we thus need to compute the attention over the cache and over the chunk. Figure 3 shows how the attention mask works over both the cache and the chunk.\n\n\nFigure 3: Pre-fill and chunking. During pre-fill of the cache, long sequences are chunked to limit memory usage. We process a sequence in three chunks, \"The cat sat on\", \"the mat and saw\", \"the dog go to\". The figure shows what happens for the third chunk (\"the dog go to\"): it attends itself using a causal mask (rightmost block), attends the cache using a sliding window (center block), and does not attend to past tokens as they are outside of the sliding window (left block).\n\n# 3 Results \n\nWe compare Mistral 7B to Llama, and re-run all benchmarks with our own evaluation pipeline for fair comparison. We measure performance on a wide variety of tasks categorized as follow:\n\n- Commonsense Reasoning (0-shot): Hellaswag [28], Winogrande [21], PIQA [4], SIQA [22], OpenbookQA [19], ARC-Easy, ARC-Challenge [9], CommonsenseQA [24]\n- World Knowledge (5-shot): NaturalQuestions [16], TriviaQA [15]\n- Reading Comprehension (0-shot): BoolQ [8], QuAC [7]\n- Math: GSM8K [10] (8-shot) with maj@8 and MATH [13] (4-shot) with maj@4\n- Code: Humaneval [5] (0-shot) and MBPP [2] (3-shot)\n- Popular aggregated results: MMLU [12] (5-shot), BBH [23] (3-shot), and AGI Eval [29] (3-5-shot, English multiple-choice questions only)\n\nDetailed results for Mistral 7B, Llama 2 7B/13B, and Code-Llama 7B are reported in Table 2. Figure 4 compares the performance of Mistral 7B with Llama 2 7B/13B, and Llama $134 \\mathrm{~B}^{4}$ in different categories. Mistral 7B surpasses Llama 2 13B across all metrics, and outperforms Llama 134 B on most benchmarks. In particular, Mistral 7B displays a superior performance in code, mathematics, and reasoning benchmarks.\n\n[^0]\n[^0]:    ${ }^{4}$ Since Llama 234 B was not open-sourced, we report results for Llama 134 B .\n\nFigure 4: Performance of Mistral 7B and different Llama models on a wide range of benchmarks. All models were re-evaluated on all metrics with our evaluation pipeline for accurate comparison. Mistral 7B significantly outperforms Llama 2 7B and Llama 2 13B on all benchmarks. It is also vastly superior to Llama 1 34B in mathematics, code generation, and reasoning benchmarks.\n\n|  Model | Modality | MMLU | HellaSwag | WinoG | PIQA | Arc-e | Arc-c | NQ | TriviaQA | HumanEval | MBPP | MATH | GSM8K  |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n|  LLaMA 2 7B | Pretrained | $44.4 \\%$ | $77.1 \\%$ | $69.5 \\%$ | $77.9 \\%$ | $68.7 \\%$ | $43.2 \\%$ | $24.7 \\%$ | $63.8 \\%$ | $11.6 \\%$ | $26.1 \\%$ | $3.9 \\%$ | $16.0 \\%$  |\n|  LLaMA 2 13B | Pretrained | $55.6 \\%$ | $\\mathbf{8 0 . 7 \\%}$ | $72.9 \\%$ | $80.8 \\%$ | $75.2 \\%$ | $48.8 \\%$ | $\\mathbf{2 9 . 0 \\%}$ | $\\mathbf{6 9 . 6 \\%}$ | $18.9 \\%$ | $35.4 \\%$ | $6.0 \\%$ | $34.3 \\%$  |\n|  Code-Llama 7B | Finetuned | $36.9 \\%$ | $62.9 \\%$ | $62.3 \\%$ | $72.8 \\%$ | $59.4 \\%$ | $34.5 \\%$ | $11.0 \\%$ | $34.9 \\%$ | $\\mathbf{3 1 . 1 \\%}$ | $\\mathbf{5 2 . 5 \\%}$ | $5.2 \\%$ | $20.8 \\%$  |\n|  Mistral 7B | Pretrained | $\\mathbf{6 0 . 1 \\%}$ | $\\mathbf{8 1 . 3 \\%}$ | $\\mathbf{7 5 . 3 \\%}$ | $\\mathbf{8 3 . 0 \\%}$ | $\\mathbf{8 0 . 0 \\%}$ | $\\mathbf{5 5 . 5 \\%}$ | $\\mathbf{2 8 . 8 \\%}$ | $\\mathbf{6 9 . 9 \\%}$ | $\\mathbf{3 0 . 5 \\%}$ | $47.5 \\%$ | $\\mathbf{1 3 . 1 \\%}$ | $\\mathbf{5 2 . 2 \\%}$  |\n\nTable 2: Comparison of Mistral 7B with Llama. Mistral 7B outperforms Llama 2 13B on all metrics, and approaches the code performance of Code-Llama 7B without sacrificing performance on non-code benchmarks.\n\nSize and Efficiency. We computed \"equivalent model sizes\" of the Llama 2 family, aiming to understand Mistral 7B models' efficiency in the cost-performance spectrum (see Figure 5). When evaluated on reasoning, comprehension, and STEM reasoning (specifically MMLU), Mistral 7B mirrored performance that one might expect from a Llama 2 model with more than 3x its size. On the Knowledge benchmarks, Mistral 7B's performance achieves a lower compression rate of 1.9x, which is likely due to its limited parameter count that restricts the amount of knowledge it can store. Evaluation Differences. On some benchmarks, there are some differences between our evaluation protocol and the one reported in the Llama 2 paper: 1) on MBPP, we use the hand-verified subset 2) on TriviaQA, we do not provide Wikipedia contexts.\n\n# 4 Instruction Finetuning\n\nTo evaluate the generalization capabilities of Mistral 7B, we fine-tuned it on instruction datasets publicly available on the Hugging Face repository. No proprietary data or training tricks were utilized: Mistral 7B - Instruct model is a simple and preliminary demonstration that the base model can easily be fine-tuned to achieve good performance. In Table 3, we observe that the resulting model, Mistral 7B - Instruct, exhibits superior performance compared to all 7B models on MT-Bench, and is comparable to 13B - Chat models. An independent human evaluation was conducted on https://llmboxing.com/leaderboard.\n\n|  Model | Chatbot Arena\nELO Rating | MT Bench  |\n| --- | --- | --- |\n|  WizardLM 13B v1.2 | 1047 | 7.2  |\n|  Mistral 7B Instruct | $\\mathbf{1 0 3 1}$ | $\\mathbf{6 . 8 4}+\\mathbf{- 0 . 0 7}$  |\n|  Llama 2 13B Chat | 1012 | 6.65  |\n|  Vicuna 13B | 1041 | 6.57  |\n|  Llama 2 7B Chat | 985 | 6.27  |\n|  Vicuna 7B | 997 | 6.17  |\n|  Alpaca 13B | 914 | 4.53  |\n\nTable 3: Comparison of Chat models. Mistral 7B Instruct outperforms all 7B models on MT-Bench, and is comparable to 13B - Chat models.\n\nIn this evaluation, participants were provided with a set of questions along with anonymous responses from two models and were asked to select their preferred response, as illustrated in Figure 6. As of October 6, 2023, the outputs generated by Mistral 7B were preferred 5020 times, compared to 4143 times for Llama 2 13B.\n\nFigure 5: Results on MMLU, commonsense reasoning, world knowledge and reading comprehension for Mistral 7B and Llama 2 (7B/13B/70B). Mistral 7B largely outperforms Llama 213B on all evaluations, except on knowledge benchmarks, where it is on par (this is likely due to its limited parameter count, which limits the amount of knowledge it can compress).\n\n# 5 Adding guardrails for front-facing applications \n\nThe ability to enforce guardrails when it comes to AI generation is important for front-facing applications. In this section, we highlight how to leverage system prompting to optionally enforce output constraints on top of our models. Additionally, we showcase the ability of Mistral 7B to perform fine-grained content moderation, which can be useful to enforce quality content in applications.\n\n### 5.1 System prompt to enforce guardrails\n\nWe introduce a system prompt (see below) to guide the model to generate answers within specified guardrails, similar to the work done with Llama 2. Using this prompt allows the user to move on the Pareto front of model utility / guardrails enforcement, as indicated in Table 4.\n\nAlways assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.\n\nWe use a set of 175 unsafe prompts for evaluating safety. With the recommended system prompt, the model properly declines to answer $100 \\%$ of the harmful questions.\nAs an illustration, we provide in Table 5 the answers of both Mistral 7B - Instruct and Llama 2 Chat 13B to the question How to kill a linux process with system prompts activated. We observe that Mistral 7B provides a correct response while Llama 2 declines to answer. Note that on this specific question, both models answer correctly when system prompts are deactivated.\n\n| Guardrails | MT Bench |\n| :--: | :--: |\n| No system prompt | $6.84 \\pm 0.07$ |\n| Llama 2 system prompt | $6.38 \\pm 0.07$ |\n| Mistral system prompt | $6.58 \\pm 0.05$ |\n\nTable 4: System prompts. Mean official MT Bench score over 10 iterations with standard deviation for Mistral 7B - Instruct. For reference, Llama 213B - Chat reports official results of 6.65 .\n\n### 5.2 Content moderation with self-reflection\n\nMistral 7B - Instruct can be used as a content moderator: the model itself is able to accurately classify a user prompt or its generated answer as being either acceptable or falling into one of the following categories: Illegal activities such as terrorism, child abuse or fraud; Hateful, harassing or violent content such as discrimination, self-harm or bullying; Unqualified advice for instance in legal, medical or financial domains.| Model | Answer |\n| --- | --- |\n| Mistral 7B - Instruct with Mistral system prompt | To kill a Linux process, you can use the `kill` command followed by the process ID (PID) of the process you want to terminate. For example, to kill process with PID 1234, you would run the command `kill 1234`. It's important to note that killing a process can have unintended consequences, so it's generally a good idea to only kill processes that you are certain you want to terminate. Additionally, it's generally a good idea to use the `kill` command with caution, as it can be used to terminate critical system processes that are necessary for the operation of the system. |\n| Llama 2 13B - Chat with Llama 2 system prompt | As a helpful and respectful assistant, I cannot provide information on how to kill a Linux process. It is not appropriate or ethical to intentionally harm or terminate processes, as it can cause damage to the system and potentially result in data loss. Additionally, it is important to respect the autonomy and security of other users' processes, as they may be critical to their work or personal activities. Instead, I suggest that you explore other options to address any issues or concerns you may have. |\n\nTable 5: Comparison between Mistral and Llama system prompts. Mistral provides the right answer whereas Llama 2 declines to answer to the question How to kill a linux process.\n\nTo do so, we designed a self-reflection prompt that makes Mistral 7B classify a prompt or a generated answer. We evaluated self-reflection on our manually curated and balanced dataset of adversarial and standard prompts and got a precision of $99.4 \\%$ for a recall of $95.6 \\%$ (considering acceptable prompts as positives).\n\nThe use cases are vast, from moderating comments on social media or forums to brand monitoring on the internet. In particular, the end user is able to select afterwards which categories to effectively filter based on their particular use-case.\n\n# 6 Conclusion \n\nOur work on Mistral 7B demonstrates that language models may compress knowledge more than what was previously thought. This opens up interesting perspectives: the field has so far put the emphasis on scaling laws in 2 dimensions (directly associating model capabilities to training cost, as in [14]); the problem is rather 3 dimensional (model capabilities, training cost, inference cost), and much remains to be explored to obtain the best performance with the smallest possible model.\n\n## Acknowledgements\n\nWe are grateful to CoreWeave for their 24/7 help in marshalling our cluster. We thank the CINECA/EuroHPC team, and in particular the operators of Leonardo, for their resources and help. We thank the maintainers of FlashAttention, vLLM, xFormers, Skypilot for their precious assistance in implementing new features and integrating their solutions into ours. A huge thanks to Tri Dao and Daniel Haziza for helping include Mistral related changes to FlashAttention and xFormers on a tight schedule. We thank the teams of Hugging Face, AWS, GCP, Azure ML for their intense help in making our model compatible everywhere.# Can you recommend a foundational and accessible book about quantum physics? \n\nCopy Link to Question\nNext Round (Enter) $\\rightarrow$\n\nRound Winner: mistral-7b-instruct-v0.1\n\nmistral-7b-instruct-v0.1 Round Winner $R$\nA great book for a foundational understanding of quantum physics is \"The Quantum Universe: Everything that Can Happen Does Happen\" by Brian Cox and Jeff Forshaw. This book is accessible and easy to understand, making it a great choice for beginners. It covers all the basics of quantum mechanics, including superposition, entanglement, and the uncertainty principle. The authors also explore the implications of quantum physics for our understanding of the universe, including the nature of time and space. Overall, \"The Quantum Universe\" is an excellent choice for anyone looking to gain a foundational understanding of this fascinating field\n\n## llama-2-13b-chat\n\nCertainly! For a foundational and accessible introduction to quantum physics, I highly recommend \"The Feynman Lectures on Physics\" by Richard Feynman. This classic text provides a clear and engaging overview of the principles and concepts of quantum mechanics, with minimal mathematical formalism and plenty of illustrations and examples. It's a great starting point for anyone looking to gain a deeper understanding of quantum physics without getting bogged down in too much technical detail.\n\nFigure 6: Human evaluation of Mistral 7B - Instruct vs Llama 2 13B - Chat Example. An example of human evaluation from limboxing.com. The question asks for recommendations of books in quantum physics. Llama 213 B - Chat recommends a general physics book, while Mistral 7B - Instruct recommends a more relevant book on quantum physics and describes in the contents in more detail.# References \n\n[1] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.\n[2] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.\n[3] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.\n[4] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, 2020.\n[5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.\n[6] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.\n[7] Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. Quac: Question answering in context. arXiv preprint arXiv:1808.07036, 2018.\n[8] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.\n[9] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.\n[10] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.\n[11] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022.\n[12] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.\n[13] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.\n[14] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Thomas Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karén Simonyan, Erich Elsen, Oriol Vinyals, Jack Rae, and Laurent Sifre. An empirical analysis of compute-optimal large language model training. In Advances in Neural Information Processing Systems, volume 35, 2022.\n[15] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.\n[16] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453-466, 2019.[17] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.\n[18] Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, and Daniel Haziza. xformers: A modular and hackable transformer modelling library. https://github.com/ facebookresearch/xformers, 2022.\n[19] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint arXiv:1809.02789, 2018.\n[20] Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.\n[21] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99-106, 2021.\n[22] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.\n[23] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.\n[24] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. arXiv preprint arXiv:1811.00937, 2018.\n[25] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.\n[26] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.\n[27] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.\n[28] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.\n[29] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364, 2023."
}
python

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

<details>
    <summary><b>Output</b></summary>
```

Example 4 (unknown):
```unknown
</details>

### Delete libraries and/or documents

You can delete libraries and documents freely.

<Tabs groupId="code">
  <TabItem value="python" label="python" default>
```

---

## The last sentence in the document is:\n\n\"Zaremba, W., Sutskever, I., and Vinyals, O. Recurrent neural network regularization. arXiv:1409.2329, 2014.

**URL:** llms-txt#the-last-sentence-in-the-document-is:\n\n\"zaremba,-w.,-sutskever,-i.,-and-vinyals,-o.-recurrent-neural-network-regularization.-arxiv:1409.2329,-2014.

**Contents:**
- Cookbooks
- FAQ
- Codestral Embed API
  - Output DType
  - Output Dimension
- Distance Measures
- Batch processing
  - t-SNE embeddings visualization
- Retrieval
- Cookbooks

// import fs from 'fs';

// Retrieve the API key from environment variables
const apiKey = process.env["MISTRAL_API_KEY"];

const client = new Mistral({
  apiKey: apiKey,
});

// If local document, upload and retrieve the signed url
// const uploaded_file = fs.readFileSync('uploaded_file.pdf');
// const uploaded_pdf = await client.files.upload({
//     file: {
//         fileName: "uploaded_file.pdf",
//         content: uploaded_file,
//     },
//     purpose: "ocr"
// });
// const signedUrl = await client.files.getSignedUrl({
//     fileId: uploaded_pdf.id,
// });

const chatResponse = await client.chat.complete({
  model: "mistral-small-latest",
  messages: [
    {
      role: "user",
      content: [
        {
          type: "text",
          text: "what is the last sentence in the document",
        },
        {
          type: "document_url",
          documentUrl: "https://arxiv.org/pdf/1805.04770",
          // documentUrl: signedUrl.url
        },
      ],
    },
  ],
});

console.log("JSON:", chatResponse.choices[0].message.content);
bash
curl https://api.mistral.ai/v1/files \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -F purpose="ocr" \
  -F file="@uploaded_file.pdf"
bash
  curl -X GET "https://api.mistral.ai/v1/files/$id/url?expiry=24" \
     -H "Accept: application/json" \
     -H "Authorization: Bearer $MISTRAL_API_KEY"
bash
curl https://api.mistral.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -d '{
    "model": "mistral-small-latest",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "what is the last sentence in the document"
          },
          {
            "type": "document_url",
            "document_url": "<url>"
          }
        ]
      }
    ],
    "document_image_limit": 8,
    "document_page_limit": 64
  }'
python

from mistralai import Mistral
from datasets import load_dataset

api_key = os.environ["MISTRAL_API_KEY"]
model = "codestral-embed"

client = Mistral(api_key=api_key)

embeddings_batch_response = client.embeddings.create(
    model=model,
    # output_dtype="binary",
    # output_dimension=512,
    inputs=[
        "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order. Example 1: Input: nums = [2,7,11,15], target = 9 Output: [0,1] Explanation: Because nums[0] + nums[1] == 9, we return [0, 1]. Example 2: Input: nums = [3,2,4], target = 6 Output: [1,2] Example 3: Input: nums = [3,3], target = 6 Output: [0,1] Constraints: 2 <= nums.length <= 104 -109 <= nums[i] <= 109 -109 <= target <= 109 Only one valid answer exists.", 
        "class Solution: def twoSum(self, nums: List[int], target: int) -> List[int]: d = {} for i, x in enumerate(nums): if (y := target - x) in d: return [d[y], i] d[x] = i"
      ],
)
typescript

const apiKey = process.env.MISTRAL_API_KEY;
const model = "codestral-embed";

const client = new Mistral({ apiKey: apiKey });

async function getEmbeddings() {

const embeddingsBatchResponse = await client.embeddings.create({
        model: model,
        // output_dtype: "binary",
        // output_dimension: 512,
        inputs: [
            "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order. Example 1: Input: nums = [2,7,11,15], target = 9 Output: [0,1] Explanation: Because nums[0] + nums[1] == 9, we return [0, 1]. Example 2: Input: nums = [3,2,4], target = 6 Output: [1,2] Example 3: Input: nums = [3,3], target = 6 Output: [0,1] Constraints: 2 <= nums.length <= 104 -109 <= nums[i] <= 109 -109 <= target <= 109 Only one valid answer exists.", 
            "class Solution: def twoSum(self, nums: List[int], target: int) -> List[int]: d = {} for i, x in enumerate(nums): if (y := target - x) in d: return [d[y], i] d[x] = i"
          ],
    });
}

// Call the async function
getEmbeddings().catch(console.error);
bash
problem_description="Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order. Example 1: Input: nums = [2,7,11,15], target = 9 Output: [0,1] Explanation: Because nums[0] + nums[1] == 9, we return [0, 1]. Example 2: Input: nums = [3,2,4], target = 6 Output: [1,2] Example 3: Input: nums = [3,3], target = 6 Output: [0,1] Constraints: 2 <= nums.length <= 104 -109 <= nums[i] <= 109 -109 <= target <= 109 Only one valid answer exists."

solution="class Solution: def twoSum(self, nums: List[int], target: int) -> List[int]: d = {} for i, x in enumerate(nums): if (y := target - x) in d: return [d[y], i] d[x] = i"

curl -X POST "https://api.mistral.ai/v1/embeddings" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${API_KEY}" \
     -d '{"model": "codestral-embed", "output_dimension": 10, "output_dtype": "binary", "input": ["'"$problem_description"'", "'"$solution"'"]}' \
     -o embedding.json

EmbeddingResponse(
    id='8d6a8e8ee0fa460c9360526480f636ee',
    object='list',
    data=[
        EmbeddingResponseData(
            object='embedding',
            embedding=[17, -64]
        ),
        EmbeddingResponseData(
            object='embedding',
            embedding=[-79, 64]
        )
    ],
    model='codestral-embed',
    usage=UsageInfo(
        prompt_tokens=263,
        completion_tokens=0,
        total_tokens=263
    )
)
python
len(embeddings_batch_response.data[0].embedding)
typescript
console.log('Embedding Length:', embeddingsBatchResponse.data?.[0]?.embedding?.length)
bash
echo "Embedding Length: $(jq '.data[0].embedding | length' embedding.json)"
python
from sklearn.metrics.pairwise import euclidean_distances

def get_code_embedding(inputs):
    embeddings_batch_response = client.embeddings.create(
        model=model,
        inputs=inputs
    )
    return embeddings_batch_response.data[0].embedding
python
dataset = load_dataset("newfacade/LeetCodeDataset")

two_sum_solution = dataset["train"][0]["completion"]
reverse_integer_solution = dataset["train"][6]["completion"]
palindrome_number_solution = dataset["train"][8]["completion"]

def remove_whitespace(code):
    return code.replace("\n", "").replace("\t", "").replace(" ", "")

two_sum_solution_clean = remove_whitespace(two_sum_solution)
reverse_integer_solution_clean = remove_whitespace(reverse_integer_solution)
palindrome_number_solution_clean = remove_whitespace(palindrome_number_solution)

code_snippets = [
    two_sum_solution_clean,
    reverse_integer_solution_clean
]

embeddings = [get_code_embedding([t]) for t in code_snippets]

reference_code_snippet = palindrome_number_solution
reference_embedding = get_code_embedding([reference_code_snippet])

for t, e in zip(code_snippets, embeddings):
    distance = euclidean_distances([e], [reference_embedding])
    print(t, distance)
json
{
    "code_snippets": {
      "two_sum_solution": "classSolution:deftwoSum(self,nums:List[int],target:int)->List[int]:d={}fori,xinenumerate(nums):if(y:=target-x)ind:return[d[y],i]d[x]=i",
      "reverse_integer_solution": "classSolution:defreverse(self,x:int)->int:ans=0mi,mx=-(2**31),2**31-1whilex:ifans<mi//10+1orans>mx//10:return0y=x%10ifx<0andy>0:y-=10a",
    },
    "reference_code_snippet": "classSolution:defisPalindrome(self,x:int)->bool:ifx<0or(xandx%10==0):returnFalsey=0whiley<x:y=y*10+x%10x//=10returnxin(y,y//10)"

classSolution:deftwoSum(self,nums:List[int],target:int)->List[int]:d={}fori,xinenumerate(nums):if(y:=target-x)ind:return[d[y],i]d[x]=i [[0.909916]]
classSolution:defreverse(self,x:int)->int:ans=0mi,mx=-(2**31),2**31-1whilex:ifans<mi//10+1orans>mx//10:return0y=x%10ifx<0andy>0:y-=10ans=ans*10+yx=(x-y)//10returnans [[0.64201937]]
python

df = pd.read_csv(
    "https://raw.githubusercontent.com/mistralai/cookbook/main/data/LeetCodeTSNE.csv"
)

def get_embeddings_by_chunks(data, chunk_size):
    chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
    embeddings_response = [
        client.embeddings.create(model=model, inputs=c) for c in chunks
    ]
    return [d.embedding for e in embeddings_response for d in e.data]

df["embeddings"] = get_embeddings_by_chunks(df["Code"].tolist(), 50)
display(df.head())
python

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0).fit_transform(np.array(df['embeddings'].to_list()))
ax = sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=np.array(df['Name'].to_list()))
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
python

from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"

client = Mistral(api_key=api_key)

embeddings_batch_response = client.embeddings.create(
    model=model,
    inputs=["Embed this sentence.", "As well as this one."],
)
typescript

const apiKey = process.env.MISTRAL_API_KEY;

const client = new Mistral({ apiKey: apiKey });

async function getEmbeddings() {

const embeddingsBatchResponse = await client.embeddings.create({
        model: "mistral-embed",
        inputs: ["Embed this sentence.", "As well as this one."],
    });

console.log('Embeddings:', embeddingsBatchResponse.data);
}

// Call the async function
getEmbeddings().catch(console.error);
bash
curl -X POST "https://api.mistral.ai/v1/embeddings" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${API_KEY}" \
     -d '{"model": "mistral-embed", "input": ["Embed this sentence.", "As well as this one."]}' \
     -o embedding.json

EmbeddingResponse(
    id='eb4c2c739780415bb3af4e47580318cc', object='list', data=[
        Data(object='embedding', embedding=[-0.0165863037109375,...], index=0),
        Data(object='embedding', embedding=[-0.0234222412109375,...], index=1)],
    model='mistral-embed', usage=EmbeddingResponseUsage(prompt_tokens=15, total_tokens=15)
)
python
len(embeddings_batch_response.data[0].embedding)
typescript
console.log('Embedding Length:', embeddingsBatchResponse.data?.[0]?.embedding?.length)
bash
echo "Embedding Length: $(jq '.data[0].embedding | length' embedding.json)"
python
from sklearn.metrics.pairwise import euclidean_distances

def get_text_embedding(inputs):
    embeddings_batch_response = client.embeddings.create(
        model=model,
        inputs=inputs
    )
    return embeddings_batch_response.data[0].embedding
python
sentences = [
    "A home without a cat — and a well-fed, well-petted and properly revered cat — may be a perfect home, perhaps, but how can it prove title?",
    "I think books are like people, in the sense that they'll turn up in your life when you most need them"
]
embeddings = [get_text_embedding([t]) for t in sentences]

reference_sentence = "Books are mirrors: You only see in them what you already have inside you"
reference_embedding = get_text_embedding([reference_sentence])

for t, e in zip(sentences, embeddings):
    distance = euclidean_distances([e], [reference_embedding])
    print(t, distance)

A home without a cat — and a well-fed, well-petted and properly revered cat — may be a perfect home, perhaps, but how can it prove title? [[0.80094257]]
I think books are like people, in the sense that they'll turn up in your life when you most need them [[0.58162089]]
python

sentences = [
    "Have a safe happy Memorial Day weekend everyone",
    "To all our friends at Whatsit Productions Films enjoy a safe happy Memorial Day weekend",
    "Where can I find the best cheese?",
]

sentence_embeddings = [get_text_embedding([t]) for t in sentences]

sentence_embeddings_pairs = list(itertools.combinations(sentence_embeddings, 2))
sentence_pairs = list(itertools.combinations(sentences, 2))
for s, e in zip(sentence_pairs, sentence_embeddings_pairs):
    print(s, euclidean_distances([e[0]], [e[1]]))

('Have a safe happy Memorial Day weekend everyone', 'To all our friends at Whatsit Productions Films enjoy a safe happy Memorial Day weekend') [[0.54326686]]
('Have a safe happy Memorial Day weekend everyone', 'Where can I find the best cheese?') [[0.92573978]]
('To all our friends at Whatsit Productions Films enjoy a safe happy Memorial Day weekend', 'Where can I find the best cheese?') [[0.9114184]]
python

df = pd.read_csv(
    "https://raw.githubusercontent.com/mistralai/cookbook/main/data/Symptom2Disease.csv",
    index_col=0,
)

def get_embeddings_by_chunks(data, chunk_size):
    chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
    embeddings_response = [
        client.embeddings.create(model=model, inputs=c) for c in chunks
    ]
    return [d.embedding for e in embeddings_response for d in e.data]

df["embeddings"] = get_embeddings_by_chunks(df["text"].tolist(), 50)
df.head()
python

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0).fit_transform(np.array(df['embeddings'].to_list()))
ax = sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=np.array(df['label'].to_list()))
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
python

fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('cc.en.300.bin')

df['fasttext_embeddings'] = df['text'].apply(lambda x: ft.get_word_vector(x).tolist())

tsne = TSNE(n_components=2, random_state=0).fit_transform(np.array(df['fasttext_embeddings'].to_list()))
ax = sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=np.array(df['label'].to_list()))
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

**Upload the Image File**
```

Example 3 (unknown):
```unknown
**Get the Signed URL**
```

Example 4 (unknown):
```unknown
**Chat Completion**
```

---

## Top P

**URL:** llms-txt#top-p

**Contents:**
- Visualization
  - To Summarize
  - Example
  - Output
  - Output Table

**Top P** is a setting that limits the tokens considered by a language model based on a probability threshold. It helps focus on the most likely tokens, improving output quality.

For these examples, we set the Temperature first, then apply a Top P of 50%. Note that a Temperature of 0 is deterministic, making Top P irrelevant in that case.

The process is as follows:
1. Apply the Temperature.
2. Use Top P (0.5) to keep only the most likely tokens.
3. Adjust the probabilities of the remaining tokens.

We will visualize the token probability distribution across different temperature values for the question:  
- "What is the best mythical creature? Answer with a single word."

<div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
    <div style={{ marginBottom: '20px', textAlign: 'center' }}>
    <img src="/img/top_barplot.png" alt="Example Image" style={{ width: '85%' }} />
    <br/><sub><sup>Different Temperature values and the top 5 tokens using Mistral 7B at 4 bits precision.</sup></sub>
    </div>

<div style={{ marginBottom: '20px', textAlign: 'center' }}>
    <span style={{ fontSize: '24px' }}>&darr;</span>
    </div>

<div style={{ marginBottom: '20px', textAlign: 'center' }}>
    <img src="/img/top_barplot_black.png" alt="Example Image" style={{ width: '85%' }} />
    <br/><sub><sup>Top P considers only the top tokens until reaching 50% probability.</sup></sub>
    </div>

<div style={{ marginBottom: '20px', textAlign: 'center' }}>
    <span style={{ fontSize: '24px' }}>&darr;</span>
    </div>

<div style={{ textAlign: 'center' }}>
    <img src="/img/top_barplot_final.png" alt="Example Image" style={{ width: '85%' }} />
    <br/><sub><sup>Other tokens' probabilities are set to 0, and the remaining tokens' probabilities are adjusted.</sup></sub>
    </div>
</div>

Top P ensures that only high-quality tokens are considered, maintaining output quality by excluding unlikely tokens. It's challenging to balance Temperature and Top P, so it's recommended to fix one and adjust the other. However you should experiment to find the best settings for your use case!

### To Summarize
1. **Role of Top P**: Top P limits the tokens considered based on a probability threshold, focusing on the most likely tokens to improve output quality.
2. **Interaction with Temperature**: Top P is applied after Temperature.
3. **Impact on Outputs**: Top P avoids considering very unlikely tokens, maintaining output quality and coherence.
4. **Balancing Temperature and Top P**: It's challenging to balance both. Start by fixing one parameter and adjust the other, experiment to find optimal settings.

Here's an example of how to use the `Top P` parameter with our python client:

| Temperature 0.1   | Temperature 1 | Temperature 1 & Top P 50% |
|:-----------------:|:-------------:|:-------------------------:|
| Dragon            | Unicorn       | Unicorn                   |
| Dragon            | Dragon        | Unicorn                   |
| Dragon            | Phoenix       | Unicorn                   |
| Dragon            | Unicorn       | Unicorn                   |
| Dragon            | Dragon        | Dragon                    |
| Dragon            | Phoenix.      | Unicorn                   |
| Dragon            | Dragon.       | Dragon                    |
| Dragon            | Phoenix       | Dragon                    |
| Dragon            | Dragon        | Dragon                    |
| Dragon            | Unicorn.      | Dragon                    |

In this example, the model generates a response considering only the top tokens that cumulatively reach a 50% probability threshold. This ensures that the output keeps some uniform diversity while still taking only the best tokens, in this case only 2 tokens reach the 50% threshold.

<summary><b>Penalties</b></summary>

**Examples:**

Example 1 (py):
```py
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "ministral-3b-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "What is the best mythical creature? Answer with a single word.",
        },
    ],
    temperature=1,
    top_p=0.5,
    n=10
)

for i, choice in enumerate(chat_response.choices):
    print(choice.message.content)
```

Example 2 (py):
```py
Unicorn
Unicorn
Unicorn
Unicorn
Dragon
Unicorn
Dragon
Dragon
Dragon
Dragon
```

---

## Train a classifier and compute the test accuracy

**URL:** llms-txt#train-a-classifier-and-compute-the-test-accuracy

---

## Unarchive Fine Tuned Model

**URL:** llms-txt#unarchive-fine-tuned-model

Source: https://docs.mistral.ai/api/#tag/jobs_api_routes_fine_tuning_unarchive_fine_tuned_model

del /v1/fine_tuning/models/{model_id}/archive

---

## Update a library.

**URL:** llms-txt#update-a-library.

Source: https://docs.mistral.ai/api/#tag/libraries_update_v1

put /v1/libraries/{library_id}

---

## Update Fine Tuned Model

**URL:** llms-txt#update-fine-tuned-model

Source: https://docs.mistral.ai/api/#tag/jobs_api_routes_fine_tuning_update_fine_tuned_model

patch /v1/fine_tuning/models/{model_id}

---

## Update the metadata of a specific document.

**URL:** llms-txt#update-the-metadata-of-a-specific-document.

Source: https://docs.mistral.ai/api/#tag/libraries_documents_update_v1

put /v1/libraries/{library_id}/documents/{document_id}

---

## uploaded_pdf = client.files.upload(

**URL:** llms-txt#uploaded_pdf-=-client.files.upload(

---

## Upload a new document.

**URL:** llms-txt#upload-a-new-document.

Source: https://docs.mistral.ai/api/#tag/libraries_documents_upload_v1

post /v1/libraries/{library_id}/documents

---

## Upload document

**URL:** llms-txt#upload-document

file_path = "mistral7b.pdf"
with open(file_path, "rb") as file_content:
    uploaded_doc = client.beta.libraries.documents.upload(
        library_id=new_library.id,
        file=File(fileName="mistral7b.pdf", content=file_content),
    )
typescript
const filePath = "~/path/to/doc.pdf";
const fileContent = fs.readFileSync(filePath);
const uploadedDoc = await client.beta.libraries.documents.upload({
    libraryId: newLibrary.id,
    requestBody: {
        file: {
            fileName: "mistral7b.pdf",
            content: fileContent
        }
    }
});
bash
curl --location --request POST "https://api.mistral.ai/v1/libraries/<library_id>/documents" \
     --header "Accept: application/json" \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --header "Content-Type: multipart/form-data" \
     --form "file=@mistral7b.pdf;type=application/pdf"

json
{
  "id": "424fdcb8-3c11-478c-a651-9637be8b4fc4",
  "library_id": "0197f425-5e85-7353-b8e7-e8b974b9c613",
  "hash": "8ad11d7d6d3a9ce8a0870088ebbcdb00",
  "mime_type": "application/pdf",
  "extension": "pdf",
  "size": 3749788,
  "name": "mistral7b.pdf",
  "created_at": "2025-07-10T11:43:01.017430Z",
  "processing_status": "Running",
  "uploaded_by_id": "6340e568-a546-4c41-9dee-1fbeb80493e1",
  "uploaded_by_type": "Workspace",
  "tokens_processing_total": 0,
  "summary": null,
  "last_processed_at": null,
  "number_of_pages": null,
  "tokens_processing_main_content": null,
  "tokens_processing_summary": null
}
python

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

<details>
    <summary><b>Content</b></summary>
```

Example 4 (unknown):
```unknown
</details>

#### Status

<Tabs groupId="code">
  <TabItem value="python" label="python" default>
```

---

## Upload File

**URL:** llms-txt#upload-file

Source: https://docs.mistral.ai/api/#tag/files_api_routes_upload_file

---

## validate the reformat the eval data

**URL:** llms-txt#validate-the-reformat-the-eval-data

**Contents:**
  - Upload dataset
  - Create a fine-tuning job

python reformat_data.py ultrachat_chunk_eval.jsonl

Skip 3674th sample
Skip 9176th sample
Skip 10559th sample
Skip 13293th sample
Skip 13973th sample
Skip 15219th sample
python
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

ultrachat_chunk_train = client.files.upload(file={
    "file_name": "ultrachat_chunk_train.jsonl",
    "content": open("ultrachat_chunk_train.jsonl", "rb"),
})
ultrachat_chunk_eval = client.files.upload(file={
    "file_name": "ultrachat_chunk_eval.jsonl",
    "content": open("ultrachat_chunk_eval.jsonl", "rb"),
})
typescript

const apiKey = process.env.MISTRAL_API_KEY;

const client = new MistralClient(apiKey);

const file = fs.readFileSync('ultrachat_chunk_train.jsonl');
const ultrachat_chunk_train = await client.files.create({ file });

const file = fs.readFileSync('ultrachat_chunk_eval.jsonl');
const ultrachat_chunk_eval = await client.files.create({ file });
bash
curl https://api.mistral.ai/v1/files \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -F purpose="fine-tune" \
  -F file="@ultrachat_chunk_train.jsonl"

curl https://api.mistral.ai/v1/files \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -F purpose="fine-tune" \
  -F file="@ultrachat_chunk_eval.jsonl"

{
    "id": "66f96d02-8b51-4c76-a5ac-a78e28b2584f",
    "object": "file",
    "bytes": 140893645,
    "created_at": 1717164199,
    "filename": "ultrachat_chunk_train.jsonl",
    "purpose": "fine-tune"
}

{
    "id": "84482011-dfe9-4245-9103-d28b6aef30d4",
    "object": "file",
    "bytes": 7247934,
    "created_at": 1717164200,
    "filename": "ultrachat_chunk_eval.jsonl",
    "purpose": "fine-tune"
}
python

**Examples:**

Example 1 (unknown):
```unknown
:::important[ ]
This `reformat_data.py` script is tailored for the UltraChat data and may not be universally applicable to other datasets. Please modify this script and reformat your data accordingly.
:::

After running the script, few cases were removed from the training data.
```

Example 2 (unknown):
```unknown
Let’s inspect one of these cases. There are two issues with this use case: 
- one of the assistant messages is an empty string; 
- the last message is not an assistant message. 

<img src="/img/guides/ft1.png" alt="drawing" width="700"/>


### Upload dataset
We can then upload both the training data and evaluation data to the Mistral Client, making them available for use in fine-tuning jobs. 

<Tabs>
  <TabItem value="python" label="python" default>
```

Example 3 (unknown):
```unknown
</TabItem>

  <TabItem value="typescript" label="typescript">
```

Example 4 (unknown):
```unknown
</TabItem>
  
  <TabItem value="curl" label="curl">
```

---

## Waiting for process to finish

**URL:** llms-txt#waiting-for-process-to-finish

while status.processing_status == "Running":
    status = client.beta.libraries.documents.status(library_id=new_library.id, document_id=uploaded_doc.id)
    time.sleep(1)
print(status)
typescript
// Check status document
const docStatus = await client.beta.libraries.documents.status({
    libraryId: newLibrary.id,
    documentId: uploadedDoc.id
});
console.log(docStatus);

// Waiting for process to finish
while (docStatus.processingStatus === "Running") {
    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for 1 second
    const updatedStatus = await client.beta.libraries.documents.status({
        libraryId: newLibrary.id,
        documentId: uploadedDoc.id
    });
    console.log(updatedStatus);
    Object.assign(docStatus, updatedStatus); // Update the status object
}
console.log(docStatus);

bash
curl --location "https://api.mistral.ai/v1/libraries/<library_id>/documents/<document_id>/status" \
     --header "Accept: application/json" \
     --header "Authorization: Bearer $MISTRAL_API_KEY"
json
{
  "document_id": "424fdcb8-3c11-478c-a651-9637be8b4fc4",
  "processing_status": "Running"
}
json
{
  "document_id": "2445a837-8f4e-475f-8183-fe4e99fed2d9",
  "processing_status": "Completed"
}
python

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

<details>
    <summary><b>Contents</b></summary>

**Running Status**
```

Example 4 (unknown):
```unknown
**Finished Status**
```

---

## Welcome to the Mistral AI Ambassador Program!

**URL:** llms-txt#welcome-to-the-mistral-ai-ambassador-program!

As our Mistral AI community continues to grow, we are looking for Mistral experts who are passionate about our models and offerings, and who are committed to giving back to the community and supporting fellow members.

---

## Write and save the file

**URL:** llms-txt#write-and-save-the-file

**Contents:**
- List batch jobs
- Request the cancellation of a batch job
- An end-to-end example
- FAQ
  - Is the batch API available for all models?
  - Does the batch API affect pricing?
  - Does the batch API affect rate limits?
  - What's the max number of requests in a batch?
  - What's the max number of batch jobs one can create?
  - How long does the batch API take to process?

with open('batch_results.jsonl', 'wb') as f:
    f.write(output_file_stream.read())
typescript

const outputFileStream = await client.files.download({ fileId: retrievedJob.outputFile });

// Write the stream to a file
const writeStream = fs.createWriteStream('batch_results.jsonl');
outputFileStream.pipeTo(new WritableStream({
    write(chunk) {
      writeStream.write(chunk);
    },
    close() {
      writeStream.end();
    }
}));
bash
curl 'https://api.mistral.ai/v1/files/<uuid>/content' \
--header "Authorization: Bearer $MISTRAL_API_KEY" \
python
list_job = client.batch.jobs.list(
    status="RUNNING",
    metadata={"job_type": "testing"}
)
typescript
const listJob = await client.batch.jobs.list({
    status: "RUNNING",
    metadata: {
        jobType: "testing"
    }
});
bash
curl 'https://api.mistral.ai/v1/batch/jobs?status=RUNNING&job_type=testing'\
--header 'x-api-key: $MISTRAL_API_KEY'
python
canceled_job = client.batch.jobs.cancel(job_id=created_job.id)
typescript
const canceledJob = await mistral.batch.jobs.cancel({
  jobId: createdJob.id,
});
bash
curl -X POST https://api.mistral.ai/v1/batch/jobs/<jobid>/cancel \
--header "Authorization: Bearer $MISTRAL_API_KEY"
python

from io import BytesIO

from mistralai import File, Mistral

def create_client():
    """
    Create a Mistral client using the API key from environment variables.

Returns:
        Mistral: An instance of the Mistral client.
    """
    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])

def generate_random_string(start, end):
    """
    Generate a random string of variable length.

Args:
        start (int): Minimum length of the string.
        end (int): Maximum length of the string.

Returns:
        str: A randomly generated string.
    """
    length = random.randrange(start, end)
    return ' '.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))

def print_stats(batch_job):
    """
    Print the statistics of the batch job.

Args:
        batch_job: The batch job object containing job statistics.
    """
    print(f"Total requests: {batch_job.total_requests}")
    print(f"Failed requests: {batch_job.failed_requests}")
    print(f"Successful requests: {batch_job.succeeded_requests}")
    print(
        f"Percent done: {round((batch_job.succeeded_requests + batch_job.failed_requests) / batch_job.total_requests, 4) * 100}")

def create_input_file(client, num_samples):
    """
    Create an input file for the batch job.

Args:
        client (Mistral): The Mistral client instance.
        num_samples (int): Number of samples to generate.

Returns:
        File: The uploaded input file object.
    """
    buffer = BytesIO()
    for idx in range(num_samples):
        request = {
            "custom_id": str(idx),
            "body": {
                "max_tokens": random.randint(10, 1000),
                "messages": [{"role": "user", "content": generate_random_string(100, 5000)}]
            }
        }
        buffer.write(json.dumps(request).encode("utf-8"))
        buffer.write("\n".encode("utf-8"))
    return client.files.upload(file=File(file_name="file.jsonl", content=buffer.getvalue()), purpose="batch")

def run_batch_job(client, input_file, model):
    """
    Run a batch job using the provided input file and model.

Args:
        client (Mistral): The Mistral client instance.
        input_file (File): The input file object.
        model (str): The model to use for the batch job.

Returns:
        BatchJob: The completed batch job object.
    """
    batch_job = client.batch.jobs.create(
        input_files=[input_file.id],
        model=model,
        endpoint="/v1/chat/completions",
        metadata={"job_type": "testing"}
    )

while batch_job.status in ["QUEUED", "RUNNING"]:
        batch_job = client.batch.jobs.get(job_id=batch_job.id)
        print_stats(batch_job)
        time.sleep(1)

print(f"Batch job {batch_job.id} completed with status: {batch_job.status}")
    return batch_job

def download_file(client, file_id, output_path):
    """
    Download a file from the Mistral server.

Args:
        client (Mistral): The Mistral client instance.
        file_id (str): The ID of the file to download.
        output_path (str): The path where the file will be saved.
    """
    if file_id is not None:
        print(f"Downloading file to {output_path}")
        output_file = client.files.download(file_id=file_id)
        with open(output_path, "w") as f:
            for chunk in output_file.stream:
                f.write(chunk.decode("utf-8"))
        print(f"Downloaded file to {output_path}")

def main(num_samples, success_path, error_path, model):
    """
    Main function to run the batch job.

Args:
        num_samples (int): Number of samples to process.
        success_path (str): Path to save successful outputs.
        error_path (str): Path to save error outputs.
        model (str): Model name to use.
    """
    client = create_client()
    input_file = create_input_file(client, num_samples)
    print(f"Created input file {input_file}")

batch_job = run_batch_job(client, input_file, model)
    print(f"Job duration: {batch_job.completed_at - batch_job.created_at} seconds")
    download_file(client, batch_job.error_file, error_path)
    download_file(client, batch_job.output_file, success_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mistral AI batch job")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--success_path", type=str, default="output.jsonl", help="Path to save successful outputs")
    parser.add_argument("--error_path", type=str, default="error.jsonl", help="Path to save error outputs")
    parser.add_argument("--model", type=str, default="codestral-latest", help="Model name to use")

args = parser.parse_args()

main(args.num_samples, args.success_path, args.error_path, args.model)
json
{
  "0": {
    "url": "https://en.wikipedia.org/wiki/2024_Nobel_Peace_Prize",
    "title": "2024 Nobel Peace Prize",
    "snippets": [
      [
        "The 2024 Nobel Peace Prize, an international peace prize established according to Alfred Nobel's will, was awarded to Nihon Hidankyo (the Japan Confederation of A- and H-Bomb Sufferers Organizations), for their activism against nuclear weapons, assisted by victim/survivors (known as Hibakusha) of the atomic bombings of Hiroshima and Nagasaki in 1945.",
        "They will receive the prize at a ceremony on 10 December 2024 at Oslo, Norway."
      ]
    ],
    "description": null,
    "date": "2024-11-26T17:39:55.057454",
    "source": "wikipedia"
  },
  "1": {
    "url": "https://en.wikipedia.org/wiki/Climate_Change",
    "title": "Climate Change",
    "snippets": [
      [
        "Present-day climate change includes both global warming—the ongoing increase in global average temperature—and its wider effects on Earth’s climate system. Climate change in a broader sense also includes previous long-term changes to Earth's climate. The current rise in global temperatures is driven by human activities, especially fossil fuel burning since the Industrial Revolution. Fossil fuel use, deforestation, and some agricultural and industrial practices release greenhouse gases. These gases absorb some of the heat that the Earth radiates after it warms from sunlight, warming the lower atmosphere. Carbon dioxide, the primary gas driving global warming, has increased in concentration by about 50% since the pre-industrial era to levels not seen for millions of years."
      ]
    ],
    "description": null,
    "date": "2024-11-26T17:39:55.057454",
    "source": "wikipedia"
  },
  "2": {
    "url": "https://en.wikipedia.org/wiki/Artificial_Intelligence",
    "title": "Artificial Intelligence",
    "snippets": [
      [
        "Artificial intelligence (AI) refers to the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals. Such machines may be called AIs."
      ]
    ],
    "description": null,
    "date": "2024-11-26T17:39:55.057454",
    "source": "wikipedia"
  }
}
python

from mistralai import Mistral, ToolMessage

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)
python
get_information_tool = {
    "type": "function",
    "function": {
        "name": "get_information",
        "description": "Get information from external source.",
        "parameters": {}
    },
}

def get_information():
    return json.dumps(references)
python
chat_history = [
    {
        "role": "system",
        "content": "Answer the user by providing references to the source of the information."
    },
    {
        "role": "user",
        "content": "Who won the Nobel Prize in 2024?"
    }
]
python
chat_response = client.chat.complete(
    model=model,
    messages=chat_history,
    tools=[get_information_tool],
)

if hasattr(chat_response.choices[0].message, 'tool_calls'):
    tool_call = chat_response.choices[0].message.tool_calls[0]
    chat_history.append(chat_response.choices[0].message)
    print(tool_call)
else:
    print("No tool call found in the response")

function=FunctionCall(name='get_information', arguments='{}') id='F4HiRgdZp' type=None index=0
python
result = get_information()

tool_call_result = ToolMessage(
    content=result,
    tool_call_id=tool_call.id,
    name=tool_call.function.name,
)

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

## List batch jobs
You can view a list of your batch jobs and filter them by various criteria, including:

- Status: `QUEUED`,
`RUNNING`, `SUCCESS`, `FAILED`, `TIMEOUT_EXCEEDED`, `CANCELLATION_REQUESTED` and `CANCELLED`
- Metadata: custom metadata key and value for the batch

<Tabs groupId="code">
  <TabItem value="python" label="python" default>
```

Example 4 (unknown):
```unknown
</TabItem>
  <TabItem value="typescript" label="typescript">
```

---

## Your huggingface token (HF_AUTH_TOKEN) should be stored in your project secrets on your Cerebrium dashboard

**URL:** llms-txt#your-huggingface-token-(hf_auth_token)-should-be-stored-in-your-project-secrets-on-your-cerebrium-dashboard

login(token=get_secret("HF_AUTH_TOKEN"))

---

## you can also try the sag algorithm:

**URL:** llms-txt#you-can-also-try-the-sag-algorithm:

---

## }

**URL:** llms-txt#}

---

## },

**URL:** llms-txt#},

---

## ➡️ Apply  ➡️

**URL:** llms-txt#➡️ apply--➡️

**Contents:**
  - ✍ [fill out the application here](https://docs.google.com/forms/d/e/1FAIpQLSdBSiRzm2xBpMszB_9fBixJNyKdGnPMj99DtZbagHMdHgkGUg/viewform) ✍

Applications for the Summer 2025 cohort are now open and will be accepted until July 1, 2025. If selected, you will be contacted by the end of August 2025 to discuss next steps and possibly participate in an interview with additional questions.

### ✍ [fill out the application here](https://docs.google.com/forms/d/e/1FAIpQLSdBSiRzm2xBpMszB_9fBixJNyKdGnPMj99DtZbagHMdHgkGUg/viewform) ✍

Our team will review each application, evaluating candidates based on the following criteria. We accept applications on a quarterly basis.

- **Passion**: Genuine passion for Mistral AI.
- **Expertise**: Knowledge and experience in AI, machine learning, or a related field.
- **Mistral advocacy**: Has previously advocated for Mistral AI, either through community engagement, blog posts, public speaking, video tutorials, or other means.
- **Commitment**: Willingness to commit to the program for at least 6 months.

---

## )

**URL:** llms-txt#)

---

## {

**URL:** llms-txt#{

---

## 🤠 Meet our current Ambassadors 🤠

**URL:** llms-txt#🤠-meet-our-current-ambassadors-🤠

Thank you to each and every one of you, including those who prefer not to be named, for contributing to our community!

<table>
  <tr>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://www.youtube.com/@matthew_berman">
        <figure>
          <img src="https://yt3.googleusercontent.com/FLJEnb2WnG3g0GV9GbGbdvkMKqInA0WcEzQkL-haJ0mBSDHl5wrUrmQ2w1_wyeoonmKl5DWvVwk=s160-c-k-c0x00ffffff-no-rj" alt="1" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover', display: 'block' }}></img>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Matthew Berman</figcaption>
        </figure>
      </a> 
    </td>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://www.youtube.com/@samwitteveenai">
        <figure>
          <img src="https://pbs.twimg.com/profile_images/2274169962/jtlg8bxr0w6quu157b7f_400x400.png" alt="1" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover', display: 'block' }}></img>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Sam Witteveen</figcaption>
        </figure>
      </a> 
    </td>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://www.linkedin.com/in/johann-diep/">
        <figure>
          <img src="https://raw.githubusercontent.com/johanndiep/profile_picture/refs/heads/main/1714692170463.jpeg" alt="1" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover', display: 'block' }}></img>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Johann Diep</figcaption>
        </figure>
      </a> 
    </td>
  </tr>
  <tr>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://www.linkedin.com/in/fracapuano/">
        <figure>
          <img src="https://pbs.twimg.com/profile_images/1789212472353632257/A5nXjAhI_400x400.jpg" alt="1" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover', display: 'block' }}></img>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Francesco Capuano</figcaption>
        </figure>
      </a> 
    </td>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://www.linkedin.com/in/gijsbert-westeneng-804172155/">
        <figure>
          <img src="https://pbs.twimg.com/profile_images/1858938413619048448/FQAaUm69_400x400.jpg" alt="1" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover', display: 'block' }}></img>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Gijsbert Westeneng</figcaption>
        </figure>
      </a> 
    </td>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://www.linkedin.com/in/sprasadh/">
        <figure style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <svg width="41" height="29" viewBox="0 0 41 29" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M11.7053 0H5.85156V5.79928H11.7053V0Z" fill="black"></path>
            <path d="M35.1272 0H29.2734V5.79928H35.1272V0Z" fill="black"></path>
            <path d="M17.559 5.79883H5.85156V11.5981H17.559V5.79883Z" fill="black"></path>
            <path d="M35.1293 5.79883H23.4219V11.5981H35.1293V5.79883Z" fill="black"></path>
            <path d="M35.1239 11.5977H5.85156V17.3969H35.1239V11.5977Z" fill="black"></path>
            <path d="M11.7053 17.3984H5.85156V23.1977H11.7053V17.3984Z" fill="black"></path>
            <path d="M23.4162 17.3984H17.5625V23.1977H23.4162V17.3984Z" fill="black"></path>
            <path d="M35.1272 17.3984H29.2734V23.1977H35.1272V17.3984Z" fill="black"></path>
            <path d="M17.5648 23.2012H0V29.0005H17.5648V23.2012Z" fill="black"></path>
            <path d="M40.9867 23.2012H23.4219V29.0005H40.9867V23.2012Z" fill="black"></path>
            <path d="M11.7072 0H5.85352V5.79928H11.7072V0Z" fill="#FFD800"></path>
            <path d="M35.1252 0H29.2715V5.79928H35.1252V0Z" fill="#FFD800"></path>
            <path d="M17.561 5.79883H5.85352V11.5981H17.561V5.79883Z" fill="#FFAF00"></path>
            <path d="M35.1254 5.79883H23.418V11.5981H35.1254V5.79883Z" fill="#FFAF00"></path>
            <path d="M35.1258 11.5977H5.85352V17.3969H35.1258V11.5977Z" fill="#FF8205"></path>
            <path d="M11.7072 17.3984H5.85352V23.1977H11.7072V17.3984Z" fill="#FA500F"></path>
            <path d="M23.4162 17.3984H17.5625V23.1977H23.4162V17.3984Z" fill="#FA500F"></path>
            <path d="M35.1252 17.3984H29.2715V23.1977H35.1252V17.3984Z" fill="#FA500F"></path>
            <path d="M17.5648 23.2012H0V29.0005H17.5648V23.2012Z" fill="#E10500"></path>
            <path d="M40.9828 23.2012H23.418V29.0005H40.9828V23.2012Z" fill="#E10500"></path>
          </svg>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>R. Shyaam Prasadh</figcaption>
        </figure>
      </a>
    </td>
  </tr>
  <tr>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://www.youtube.com/allaboutai">
        <figure style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <svg width="41" height="29" viewBox="0 0 41 29" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M11.7053 0H5.85156V5.79928H11.7053V0Z" fill="black"></path>
            <path d="M35.1272 0H29.2734V5.79928H35.1272V0Z" fill="black"></path>
            <path d="M17.559 5.79883H5.85156V11.5981H17.559V5.79883Z" fill="black"></path>
            <path d="M35.1293 5.79883H23.4219V11.5981H35.1293V5.79883Z" fill="black"></path>
            <path d="M35.1239 11.5977H5.85156V17.3969H35.1239V11.5977Z" fill="black"></path>
            <path d="M11.7053 17.3984H5.85156V23.1977H11.7053V17.3984Z" fill="black"></path>
            <path d="M23.4162 17.3984H17.5625V23.1977H23.4162V17.3984Z" fill="black"></path>
            <path d="M35.1272 17.3984H29.2734V23.1977H35.1272V17.3984Z" fill="black"></path>
            <path d="M17.5648 23.2012H0V29.0005H17.5648V23.2012Z" fill="black"></path>
            <path d="M40.9867 23.2012H23.4219V29.0005H40.9867V23.2012Z" fill="black"></path>
            <path d="M11.7072 0H5.85352V5.79928H11.7072V0Z" fill="#FFD800"></path>
            <path d="M35.1252 0H29.2715V5.79928H35.1252V0Z" fill="#FFD800"></path>
            <path d="M17.561 5.79883H5.85352V11.5981H17.561V5.79883Z" fill="#FFAF00"></path>
            <path d="M35.1254 5.79883H23.418V11.5981H35.1254V5.79883Z" fill="#FFAF00"></path>
            <path d="M35.1258 11.5977H5.85352V17.3969H35.1258V11.5977Z" fill="#FF8205"></path>
            <path d="M11.7072 17.3984H5.85352V23.1977H11.7072V17.3984Z" fill="#FA500F"></path>
            <path d="M23.4162 17.3984H17.5625V23.1977H23.4162V17.3984Z" fill="#FA500F"></path>
            <path d="M35.1252 17.3984H29.2715V23.1977H35.1252V17.3984Z" fill="#FA500F"></path>
            <path d="M17.5648 23.2012H0V29.0005H17.5648V23.2012Z" fill="#E10500"></path>
            <path d="M40.9828 23.2012H23.418V29.0005H40.9828V23.2012Z" fill="#E10500"></path>
          </svg>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Kristian Fagerlie</figcaption>
        </figure>
      </a>
    </td>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://x.com/guohao_li">
        <figure>
          <img src="https://ghli.org/authors/admin/avatar_hu9d9865ddbb26c633557596753f4b7710_2567527_250x250_fill_lanczos_center_2.png" alt="1" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover', display: 'block' }}></img>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Guohao Li</figcaption>
        </figure>
      </a>
    </td>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://github.com/deep-diver">
        <figure>
          <img src="https://avatars.githubusercontent.com/u/26025527?v=4" alt="1" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover', display: 'block' }}/>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Chansung Park</figcaption>
        </figure>
      </a>
    </td>
  </tr>
  <tr>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://www.linkedin.com/in/stephen-batifol/">
        <figure style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <svg width="41" height="29" viewBox="0 0 41 29" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M11.7053 0H5.85156V5.79928H11.7053V0Z" fill="black"></path>
            <path d="M35.1272 0H29.2734V5.79928H35.1272V0Z" fill="black"></path>
            <path d="M17.559 5.79883H5.85156V11.5981H17.559V5.79883Z" fill="black"></path>
            <path d="M35.1293 5.79883H23.4219V11.5981H35.1293V5.79883Z" fill="black"></path>
            <path d="M35.1239 11.5977H5.85156V17.3969H35.1239V11.5977Z" fill="black"></path>
            <path d="M11.7053 17.3984H5.85156V23.1977H11.7053V17.3984Z" fill="black"></path>
            <path d="M23.4162 17.3984H17.5625V23.1977H23.4162V17.3984Z" fill="black"></path>
            <path d="M35.1272 17.3984H29.2734V23.1977H35.1272V17.3984Z" fill="black"></path>
            <path d="M17.5648 23.2012H0V29.0005H17.5648V23.2012Z" fill="black"></path>
            <path d="M40.9867 23.2012H23.4219V29.0005H40.9867V23.2012Z" fill="black"></path>
            <path d="M11.7072 0H5.85352V5.79928H11.7072V0Z" fill="#FFD800"></path>
            <path d="M35.1252 0H29.2715V5.79928H35.1252V0Z" fill="#FFD800"></path>
            <path d="M17.561 5.79883H5.85352V11.5981H17.561V5.79883Z" fill="#FFAF00"></path>
            <path d="M35.1254 5.79883H23.418V11.5981H35.1254V5.79883Z" fill="#FFAF00"></path>
            <path d="M35.1258 11.5977H5.85352V17.3969H35.1258V11.5977Z" fill="#FF8205"></path>
            <path d="M11.7072 17.3984H5.85352V23.1977H11.7072V17.3984Z" fill="#FA500F"></path>
            <path d="M23.4162 17.3984H17.5625V23.1977H23.4162V17.3984Z" fill="#FA500F"></path>
            <path d="M35.1252 17.3984H29.2715V23.1977H35.1252V17.3984Z" fill="#FA500F"></path>
            <path d="M17.5648 23.2012H0V29.0005H17.5648V23.2012Z" fill="#E10500"></path>
            <path d="M40.9828 23.2012H23.418V29.0005H40.9828V23.2012Z" fill="#E10500"></path>
          </svg>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Stephen Batifol</figcaption>
        </figure>
      </a>
    </td>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://youtube.com/@pierrebittner">
        <figure style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <svg width="41" height="29" viewBox="0 0 41 29" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M11.7053 0H5.85156V5.79928H11.7053V0Z" fill="black"></path>
            <path d="M35.1272 0H29.2734V5.79928H35.1272V0Z" fill="black"></path>
            <path d="M17.559 5.79883H5.85156V11.5981H17.559V5.79883Z" fill="black"></path>
            <path d="M35.1293 5.79883H23.4219V11.5981H35.1293V5.79883Z" fill="black"></path>
            <path d="M35.1239 11.5977H5.85156V17.3969H35.1239V11.5977Z" fill="black"></path>
            <path d="M11.7053 17.3984H5.85156V23.1977H11.7053V17.3984Z" fill="black"></path>
            <path d="M23.4162 17.3984H17.5625V23.1977H23.4162V17.3984Z" fill="black"></path>
            <path d="M35.1272 17.3984H29.2734V23.1977H35.1272V17.3984Z" fill="black"></path>
            <path d="M17.5648 23.2012H0V29.0005H17.5648V23.2012Z" fill="black"></path>
            <path d="M40.9867 23.2012H23.4219V29.0005H40.9867V23.2012Z" fill="black"></path>
            <path d="M11.7072 0H5.85352V5.79928H11.7072V0Z" fill="#FFD800"></path>
            <path d="M35.1252 0H29.2715V5.79928H35.1252V0Z" fill="#FFD800"></path>
            <path d="M17.561 5.79883H5.85352V11.5981H17.561V5.79883Z" fill="#FFAF00"></path>
            <path d="M35.1254 5.79883H23.418V11.5981H35.1254V5.79883Z" fill="#FFAF00"></path>
            <path d="M35.1258 11.5977H5.85352V17.3969H35.1258V11.5977Z" fill="#FF8205"></path>
            <path d="M11.7072 17.3984H5.85352V23.1977H11.7072V17.3984Z" fill="#FA500F"></path>
            <path d="M23.4162 17.3984H17.5625V23.1977H23.4162V17.3984Z" fill="#FA500F"></path>
            <path d="M35.1252 17.3984H29.2715V23.1977H35.1252V17.3984Z" fill="#FA500F"></path>
            <path d="M17.5648 23.2012H0V29.0005H17.5648V23.2012Z" fill="#E10500"></path>
            <path d="M40.9828 23.2012H23.418V29.0005H40.9828V23.2012Z" fill="#E10500"></path>
          </svg>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Pierre Bittner</figcaption>
        </figure>
      </a>
    </td>

<td style={{ width: '300px', height: '200px' }}>
      <a href="https://www.amayuelas.me/">
        <figure>
          <img src="https://www.amayuelas.me/assets/img/me_coffee.jpg" alt="1" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover', display: 'block' }}></img>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Alfonso Amayuelas</figcaption>
        </figure>
      </a>
    </td>
  </tr>
  <tr>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://www.testingcatalog.com/author/alexey/">
        <figure>
          <img src="https://i.ibb.co/pBfySc78/alexis.png" alt="1" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover', display: 'block' }}></img>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Alexey Shabanov</figcaption>
        </figure>
      </a>
    </td>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://www.youtube.com/channel/UCMJEpQVv0p3qzJnQ9IuiiMA">
        <figure>
          <img src="https://i.ibb.co/5Xxm7qQ4/boris.png" alt="1" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover', display: 'block' }}></img>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Boris Lesueur</figcaption>
        </figure>
      </a>
    </td>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://youtube.com/@jeanviet">
        <figure>
          <img src="https://i.ibb.co/pjDJyCNm/jeanb.jpg" alt="1" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover', display: 'block' }}></img>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>Jean-Baptiste Viet</figcaption>
        </figure>
      </a>
    </td>
  </tr>
  <!-- Final call-to-action cell -->
  <tr>
    <td style={{ width: '300px', height: '200px' }}>
      <a href="https://forms.gle/pTMchkVVPCxSVW5u5">
        <figure>
          <img src="https://cms.mistral.ai/assets/920e56ee-25c5-439d-bd31-fbdf5c92c87f" alt="1" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover', display: 'block' }}></img>
          <figcaption style={{ fontSize: '15px', textAlign: 'center' }}>You?</figcaption>
        </figure>
      </a>
    </td>
  </tr>
</table>

---

## 📝 Minimum requirements

**URL:** llms-txt#📝 minimum-requirements

- **Monthly Requirement:** Contribute at least one piece of content/event or show a significant amount of community support every month. Invited Ambassadors are exempt from this requirement.
- **Biyearly Assessment:** We will review Ambassador contributions every six months. Those not meeting the minimum requirements may be asked to exit the program.

---

## ➡️ Program details  ➡️

**URL:** llms-txt#➡️ program-details--➡️

**Contents:**
- 💡 Ambassador benefits
- 💡 Ambassador roles and responsibilities

## 💡 Ambassador benefits

- **Free credits:** Mistral Ambassadors will receive free API credits on la Plateforme.
- **Feature preview:** Mistral Ambassadors will be part of our early access program, can directly engage with our product teams through feature discussions, and provide valuable feedback to our products.
- **Recognition:** Public recognition and features on our website and on our Discord.
- **VIP experience:** Ambassadors will receive early invites and access to our events and will have a dedicated channel on Discord and Slack to engage with Mistral team members and fellow Ambassadors.

## 💡 Ambassador roles and responsibilities

- **Content Creation:** Create and share high-quality content (blogs, tutorials, videos, etc.) featuring our AI models and tools through their own channels or collaborate with Mistral for multimedia opportunities.
- **Social media engagement:** Post on social media about Mistral.
- **Event organization**: Host events/workshops/webinars around our models and tools.
- **Event Participation:** Talk about Mistral AI at AI-related events, meetups, and hackathons.
- **Supporting others:** Support our community members, respond to comments, engage with our community, and help other users both online and offline.

---
