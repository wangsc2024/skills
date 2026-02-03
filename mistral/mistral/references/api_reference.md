# Mistral - Api Reference

**Pages:** 4

---

## "api_key": "WANDB_KEY",

**URL:** llms-txt#"api_key":-"wandb_key",

---

## Print references only

**URL:** llms-txt#print-references-only

if refs_used:
    print("\n\nSources:")
    for i, ref in enumerate(set(refs_used), 1):
        reference = json.loads(result)[str(ref)]
        print(f"\n{i}. {reference['title']}: {reference['url']}")

The Nobel Peace Prize for 2024 was awarded to the Japan Confederation of A- and H-Bomb Sufferers Organizations (Nihon Hidankyo) for their activism against nuclear weapons, including efforts by survivors of the atomic bombings of Hiroshima and Nagasaki.

1. 2024 Nobel Peace Prize: https://en.wikipedia.org/wiki/2024_Nobel_Peace_Prize
```

**Examples:**

Example 1 (unknown):
```unknown
Output:
```

---

## Print the main response and save each reference

**URL:** llms-txt#print-the-main-response-and-save-each-reference

for chunk in chat_response.choices[0].message.content:
    if isinstance(chunk, TextChunk):
        print(chunk.text, end="")
    elif isinstance(chunk, ReferenceChunk):
        refs_used += chunk.reference_ids

---

## Retrieve the API key from environment variables

**URL:** llms-txt#retrieve-the-api-key-from-environment-variables

api_key = os.environ["MISTRAL_API_KEY"]

---
