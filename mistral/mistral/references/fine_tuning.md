# Mistral - Fine Tuning

**Pages:** 10

---

## Cancel Fine Tuning Job

**URL:** llms-txt#cancel-fine-tuning-job

Source: https://docs.mistral.ai/api/#tag/jobs_api_routes_fine_tuning_cancel_fine_tuning_job

post /v1/fine_tuning/jobs/{job_id}/cancel

---

## create a fine-tuning job

**URL:** llms-txt#create-a-fine-tuning-job

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

---

## Create Fine Tuning Job

**URL:** llms-txt#create-fine-tuning-job

Source: https://docs.mistral.ai/api/#tag/jobs_api_routes_fine_tuning_create_fine_tuning_job

post /v1/fine_tuning/jobs

---

## Get Fine Tuning Jobs

**URL:** llms-txt#get-fine-tuning-jobs

Source: https://docs.mistral.ai/api/#tag/jobs_api_routes_fine_tuning_get_fine_tuning_jobs

get /v1/fine_tuning/jobs

---

## Get Fine Tuning Job

**URL:** llms-txt#get-fine-tuning-job

Source: https://docs.mistral.ai/api/#tag/jobs_api_routes_fine_tuning_get_fine_tuning_job

get /v1/fine_tuning/jobs/{job_id}

---

## "project": "finetuning",

**URL:** llms-txt#"project":-"finetuning",

---

## start a fine-tuning job

**URL:** llms-txt#start-a-fine-tuning-job

**Contents:**
  - Analyze and evaluate fine-tuned model

client.fine_tuning.jobs.start(job_id = created_jobs.id)

created_jobs
typescript
const createdJob = await client.jobs.create({
  model: 'open-mistral-7b',
  trainingFiles: [ultrachat_chunk_train.id],
  validationFiles: [ultrachat_chunk_eval.id],
  hyperparameters: {
    trainingSteps: 10,
    learningRate: 0.0001,
  },
});
bash
curl https://api.mistral.ai/v1/fine_tuning/jobs \
--header "Authorization: Bearer $MISTRAL_API_KEY" \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--data '{
  "model": "open-mistral-7b",
  "training_files": [
    "<uuid>"
  ],
  "validation_files": [
    "<uuid>"
  ],
  "hyperparameters": {
    "training_steps": 10,
    "learning_rate": 0.0001
  }
}'

{
    "id": "25d7efe6-6303-474f-9739-21fb0fccd469",
    "hyperparameters": {
        "training_steps": 10,
        "learning_rate": 0.0001
    },
    "fine_tuned_model": null,
    "model": "open-mistral-7b",
    "status": "QUEUED",
    "job_type": "FT",
    "created_at": 1717170356,
    "modified_at": 1717170357,
    "training_files": [
        "66f96d02-8b51-4c76-a5ac-a78e28b2584f"
    ],
    "validation_files": [
        "84482011-dfe9-4245-9103-d28b6aef30d4"
    ],
    "object": "job",
    "integrations": []
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

Example output:
```

Example 4 (unknown):
```unknown
### Analyze and evaluate fine-tuned model

When we retrieve a model, we get the following metrics every 10% of the progress with a minimum of 10 steps in between:
- Training loss: the error of the model on the training data, indicating how well the model is learning from the training set. 
- Validation loss: the error of the model on the validation data, providing insight into how well the model is generalizing to unseen data. 
- Validation token accuracy: the percentage of tokens in the validation set that are correctly predicted by the model. 

Both validation loss and validation token accuracy serve as essential indicators of the model's overall performance, helping to assess its ability to generalize and make accurate predictions on new data.


<Tabs>
  <TabItem value="python" label="python" default>
```

---

## Start Fine Tuning Job

**URL:** llms-txt#start-fine-tuning-job

Source: https://docs.mistral.ai/api/#tag/jobs_api_routes_fine_tuning_start_fine_tuning_job

post /v1/fine_tuning/jobs/{job_id}/start

---

## validate and reformat the training data

**URL:** llms-txt#validate-and-reformat-the-training-data

python reformat_data.py ultrachat_chunk_train.jsonl

---

## ]

**URL:** llms-txt#]

)
typescript
const createdJob = await client.fineTuning.jobs.create({
    model: 'open-mistral-7b',
    trainingFiles: [{fileId: training_data.id, weight: 1}],
    validationFiles: [validation_data.id],
    hyperparameters: {
      trainingSteps: 10,
      learningRate: 0.0001,
    },
    autoStart:false,
//  integrations=[
//      {
//          project: "finetuning",
//          apiKey: "WANDB_KEY",
//      }
//  ]
});
bash
curl https://api.mistral.ai/v1/fine_tuning/jobs \
--header "Authorization: Bearer $MISTRAL_API_KEY" \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--data '{
  "model": "open-mistral-7b",
  "training_files": [
    "<uuid>"
  ],
  "validation_files": [
    "<uuid>"
  ],
  "hyperparameters": {
    "training_steps": 10,
    "learning_rate": 0.0001
  },
  "auto_start": false
}'
bash
curl https://api.mistral.ai/v1/fine_tuning/jobs/<jobid> \
--header "Authorization: Bearer $MISTRAL_API_KEY"
python

**Examples:**

Example 1 (unknown):
```unknown
After creating a fine-tuning job, you can check the job status using
`client.fine_tuning.jobs.get(job_id = created_jobs.id)`.
  </TabItem>

  <TabItem value="typescript" label="typescript">
```

Example 2 (unknown):
```unknown
After creating a fine-tuning job, you can check the job status using
`client.fineTuning.jobs.get({ jobId: createdJob.id })`.
  </TabItem>

  <TabItem value="curl" label="curl">
```

Example 3 (unknown):
```unknown
After creating a fine-tuning job, you can check the job status using:
```

Example 4 (unknown):
```unknown
</TabItem>

</Tabs>

Initially, the job status will be `"QUEUED"`.
After a brief period, the status will update to `"VALIDATED"`.
At this point, you can proceed to start the fine-tuning job:

<Tabs groupId="code">
  <TabItem value="python" label="python" default>
```

---
