# Mistral - Audio

**Pages:** 5

---

## Encode the audio file in base64

**URL:** llms-txt#encode-the-audio-file-in-base64

with open("examples/files/bcn_weather.mp3", "rb") as f:
    content = f.read()
audio_base64 = base64.b64encode(content).decode('utf-8')

---

## Get the transcription

**URL:** llms-txt#get-the-transcription

transcription_response = client.audio.transcriptions.complete(
    model=model,
    file_url=signed_url.url,
    ## language="en"
)

---

## If local audio, upload and retrieve the signed url

**URL:** llms-txt#if-local-audio,-upload-and-retrieve-the-signed-url

with open("local_audio.mp3", "rb") as f:
    uploaded_audio = client.files.upload(
        file={
            "content": f,
            "file_name": "local_audio.mp3",
            },
        purpose="audio"
    )

signed_url = client.files.get_signed_url(file_id=uploaded_audio.id)

---

## Print the contents

**URL:** llms-txt#print-the-contents

**Contents:**
- FAQ
- Prepare and upload your batch
- Create a new batch job
- Get a batch job details
- Get batch job results

print(transcription_response)
typescript

// Retrieve the API key from environment variables
const apiKey = process.env["MISTRAL_API_KEY"];

// Initialize the Mistral client
const client = new Mistral({ apiKey: apiKey });

// Transcribe the audio with timestamps
const transcriptionResponse = await client.audio.transcriptions.complete({
  model: "voxtral-mini-latest",
  fileUrl: "https://docs.mistral.ai/audio/obama.mp3",
  timestamp_granularities: "segment"
});

// Log the contents
console.log(transcriptionResponse);
bash
curl --location 'https://api.mistral.ai/v1/audio/transcriptions' \
--header "x-api-key: $MISTRAL_API_KEY" \
--form 'file_url="https://docs.mistral.ai/audio/obama.mp3"' \
--form 'model="voxtral-mini-latest"'
--form 'timestamp_granularities="segment"'
json
{
  "model": "voxtral-mini-2507",
  "text": "This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms, in schools, at farms and on factory floors, at diners and on distant military outposts. All these conversations are what have kept me honest, kept me inspired, and kept me going. Every day, I learned from you. You made me a better President, and you made me a better man. Over the course of these eight years, I've seen the goodness, the resilience, and the hope of the American people. I've seen neighbors looking out for each other as we rescued our economy from the worst crisis of our lifetimes. I've hugged cancer survivors who finally know the security of affordable health care. I've seen communities like Joplin rebuild from disaster, and cities like Boston show the world that no terrorist will ever break the American spirit. I've seen the hopeful faces of young graduates and our newest military officers. I've mourned with grieving families searching for answers. And I found grace in a Charleston church. I've seen our scientists help a paralyzed man regain his sense of touch and our wounded warriors walk again. I've seen our doctors and volunteers rebuild after earthquakes and stop pandemics in their tracks. I've learned from students who are building robots and curing diseases and who will change the world in ways we can't even imagine. I've seen the youngest of children remind us of our obligations to care for our refugees. to work in peace, and above all, to look out for each other. That's what's possible when we come together in the slow, hard, sometimes frustrating, but always vital work of self-government. But we can't take our democracy for granted. All of us, regardless of party, should throw ourselves into the work of citizenship. Not just when there's an election. Not just when our own narrow interest is at stake. But over the full span of a lifetime. If you're tired of arguing with strangers on the Internet, try to talk with one in real life. If something needs fixing, lace up your shoes and do some organizing. If you're disappointed by your elected officials, then grab a clipboard, get some signatures, and run for office yourself. Our success depends on our participation, regardless of which way the pendulum of power swings. It falls on each of us to be guardians of our democracy. to embrace the joyous task we've been given to continually try to improve this great nation of ours. Because for all our outward differences, we all share the same proud title, citizen. It has been the honor of my life to serve you as president. Eight years later, I am even more optimistic about our country's promise, and I look forward to working along your side as a citizen for all my days that remain. Thanks, everybody. God bless you, and God bless the United States of America.",
  "language": null,
  "segments": [
    {
      "text": "This week, I traveled to Chicago to deliver my final farewell address to the nation, following",
      "start": 0.8,
      "end": 6.2
    },
    {
      "text": "in the tradition of presidents before me.",
      "start": 6.2,
      "end": 9.0
    },
    {
      "text": "It was an opportunity to say thank you.",
      "start": 9.0,
      "end": 11.8
    },
    {
      "text": "Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American",
      "start": 11.8,
      "end": 17.6
    },
    {
      "text": "people, in living rooms, in schools, at farms and on factory floors, at diners and on distant",
      "start": 17.6,
      "end": 24.9
    },
    {
      "text": "military outposts.",
      "start": 24.9,
      "end": 26.6
    },
    {
      "text": "All these conversations are what have kept me honest, kept me inspired, and kept me going.",
      "start": 26.6,
      "end": 32.8
    },
    {
      "text": "Every day, I learned from you.",
      "start": 32.8,
      "end": 35.4
    },
    {
      "text": "You made me a better President, and you made me a better man.",
      "start": 35.4,
      "end": 39.3
    },
    {
      "text": "Over the course of these eight years, I've seen the goodness, the resilience, and the hope of the American people.",
      "start": 39.3,
      "end": 46.1
    },
    {
      "text": "I've seen neighbors looking out for each other as we rescued our economy from the worst crisis of our lifetimes.",
      "start": 46.1,
      "end": 51.3
    },
    {
      "text": "I've hugged cancer survivors who finally know the security of affordable health care.",
      "start": 52.2,
      "end": 56.5
    },
    {
      "text": "I've seen communities like Joplin rebuild from disaster, and cities like Boston show the world that no terrorist will ever break the American spirit.",
      "start": 57.1,
      "end": 65.7
    },
    {
      "text": "I've seen the hopeful faces of young graduates and our newest military officers.",
      "start": 66.5,
      "end": 71.1
    },
    {
      "text": "I've mourned with grieving families searching for answers.",
      "start": 71.7,
      "end": 74.4
    },
    {
      "text": "And I found grace in a Charleston church.",
      "start": 75.2,
      "end": 77.7
    },
    {
      "text": "I've seen our scientists help a paralyzed man regain his sense of touch and our wounded warriors walk again.",
      "start": 78.5,
      "end": 85.2
    },
    {
      "text": "I've seen our doctors and volunteers rebuild after earthquakes and stop pandemics in their tracks.",
      "start": 85.9,
      "end": 91.9
    },
    {
      "text": "I've learned from students who are building robots and curing diseases and who will change the world in ways we can't even imagine.",
      "start": 92.6,
      "end": 99.2
    },
    {
      "text": "I've seen the youngest of children remind us of our obligations to care for our refugees.",
      "start": 100.1,
      "end": 105.8
    },
    {
      "text": "to work in peace, and above all, to look out for each other.",
      "start": 106.6,
      "end": 111.6
    },
    {
      "text": "That's what's possible when we come together in the slow, hard, sometimes frustrating, but always vital work of self-government.",
      "start": 111.6,
      "end": 120.3
    },
    {
      "text": "But we can't take our democracy for granted.",
      "start": 120.3,
      "end": 123.4
    },
    {
      "text": "All of us, regardless of party, should throw ourselves into the work of citizenship.",
      "start": 123.4,
      "end": 129.2
    },
    {
      "text": "Not just when there's an election.",
      "start": 129.2,
      "end": 131.2
    },
    {
      "text": "Not just when our own narrow interest is at stake.",
      "start": 131.2,
      "end": 134.7
    },
    {
      "text": "But over the full span of a lifetime.",
      "start": 134.7,
      "end": 138.1
    },
    {
      "text": "If you're tired of arguing with strangers on the Internet,",
      "start": 138.1,
      "end": 141.4
    },
    {
      "text": "try to talk with one in real life.",
      "start": 141.4,
      "end": 144.0
    },
    {
      "text": "If something needs fixing,",
      "start": 144.0,
      "end": 146.0
    },
    {
      "text": "lace up your shoes and do some organizing.",
      "start": 146.0,
      "end": 149.3
    },
    {
      "text": "If you're disappointed by your elected officials, then grab a clipboard, get some signatures, and run for office yourself.",
      "start": 149.3,
      "end": 156.8
    },
    {
      "text": "Our success depends on our participation, regardless of which way the pendulum of power swings.",
      "start": 156.8,
      "end": 165.3
    },
    {
      "text": "It falls on each of us to be guardians of our democracy.",
      "start": 165.3,
      "end": 168.5
    },
    {
      "text": "to embrace the joyous task we've been given to continually try to improve this great nation of ours.",
      "start": 168.5,
      "end": 174.6
    },
    {
      "text": "Because for all our outward differences, we all share the same proud title, citizen.",
      "start": 175.4,
      "end": 181.7
    },
    {
      "text": "It has been the honor of my life to serve you as president.",
      "start": 182.7,
      "end": 186.0
    },
    {
      "text": "Eight years later, I am even more optimistic about our country's promise,",
      "start": 186.9,
      "end": 190.3
    },
    {
      "text": "and I look forward to working along your side as a citizen for all my days that remain.",
      "start": 190.3,
      "end": 197.3
    },
    {
      "text": "Thanks, everybody. God bless you, and God bless the United States of America.",
      "start": 198.5,
      "end": 203.4
    }
  ],
  "usage": {
    "prompt_audio_seconds": 203,
    "prompt_tokens": 4,
    "total_tokens": 3945,
    "completion_tokens": 1316
  }
}
bash
{"custom_id": "0", "body": {"max_tokens": 100, "messages": [{"role": "user", "content": "What is the best French cheese?"}]}}
{"custom_id": "1", "body": {"max_tokens": 100, "messages": [{"role": "user", "content": "What is the best French wine?"}]}}
python
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

batch_data = client.files.upload(
    file={
        "file_name": "test.jsonl",
        "content": open("test.jsonl", "rb")
    },
    purpose = "batch"
)
typescript

const apiKey = process.env.MISTRAL_API_KEY;

const client = new Mistral({apiKey: apiKey});

const batchFile = fs.readFileSync('batch_input_file.jsonl');
const batchData = await client.files.upload({
    file: {
        fileName: "batch_input_file.jsonl",
        content: batchFile,
    },
    purpose: "batch"
});
curl
curl https://api.mistral.ai/v1/files \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -F purpose="batch" \
  -F file="@batch_input_file.jsonl"
python
created_job = client.batch.jobs.create(
    input_files=[batch_data.id],
    model="mistral-small-latest",
    endpoint="/v1/chat/completions",
    metadata={"job_type": "testing"}
)
typescript

const apiKey = process.env.MISTRAL_API_KEY;

const client = new Mistral({apiKey: apiKey});

const createdJob = await client.batch.jobs.create({
    inputFiles: [batchData.id],
    model: "mistral-small-latest",
    endpoint: "/v1/chat/completions",
    metadata: {jobType: "testing"}
});
bash
curl --location "https://api.mistral.ai/v1/batch/jobs" \
--header "Authorization: Bearer $MISTRAL_API_KEY" \
--header "Content-Type: application/json" \
--header "Accept: application/json" \
--data '{
    "model": "mistral-small-latest",
    "input_files": [
        "<uuid>"
    ],
    "endpoint": "/v1/chat/completions",
    "metadata": {
        "job_type": "testing"
    }
}'
python
retrieved_job = client.batch.jobs.get(job_id=created_job.id)
typescript
const retrievedJob = await client.batch.jobs.get({ jobId: createdJob.id});
bash
curl https://api.mistral.ai/v1/batch/jobs/<jobid> \
--header "Authorization: Bearer $MISTRAL_API_KEY"
python
output_file_stream = client.files.download(file_id=retrieved_job.output_file)

**Examples:**

Example 1 (unknown):
```unknown
</TabItem>
  <TabItem value="typescript" label="typescript">
```

Example 2 (unknown):
```unknown
</TabItem>
  <TabItem value="curl" label="curl" default>
```

Example 3 (unknown):
```unknown
</TabItem>
</Tabs>

<details>
<summary><b>JSON Output</b></summary>
```

Example 4 (unknown):
```unknown
</details>

## FAQ

- **What's the maximum audio length?**

    The maximum length will depend on the endpoint used, currently the limits are as follows:
    - ≈20 minutes for [Chat with Audio](#chat-with-audio) for both models.
    - ≈15 minutes for [Transcription](#transcription), longer transcriptions will be available soon.

:::tip
Here are some tips if you need to handle longer audio files:
- **Divide the audio into smaller segments:** Transcribe each segment individually. However, be aware that this might lead to a loss of context, difficulties in splitting the audio at natural pauses (such as mid-sentence), and the need to combine the transcriptions afterward.
- **Increase the playback speed:** Send the file at a faster pace by speeding up the audio. Keep in mind that this can reduce audio quality and require adjusting the transcription timestamps to align with the original audio file.
:::


[Batch Inference]
Source: https://docs.mistral.ai/docs/capabilities/batch_inference

## Prepare and upload your batch

A batch is composed of a list of API requests. The structure of an individual request includes:

- A unique `custom_id` for identifying each request and referencing results after completion
- A `body` object with message information

Here's an example of how to structure a batch request:
```

---

## Transcribe the audio with timestamps

**URL:** llms-txt#transcribe-the-audio-with-timestamps

transcription_response = client.audio.transcriptions.complete(
    model=model,
    file_url="https://docs.mistral.ai/audio/obama.mp3",
    timestamp_granularities="segment"
)

---
