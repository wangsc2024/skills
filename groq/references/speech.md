# Groq - Speech

**Pages:** 3

---

## Python code snippet for transcription

**URL:** llms-txt#python-code-snippet-for-transcription

**Examples:**

Example 1 (unknown):
```unknown
The Groq SDK package can be installed using the following command:
```

---

## Specify the path to the audio file

**URL:** llms-txt#specify-the-path-to-the-audio-file

filename = os.path.dirname(__file__) + "/sample_audio.m4a" # Replace with your audio file!

---

## Speech to Text

**URL:** llms-txt#speech-to-text

**Contents:**
- API Endpoints
- Supported Models
- Which Whisper Model Should You Use?
- Working with Audio Files
  - Audio File Limitations
  - Audio Preprocessing
  - Working with Larger Audio Files
- Using the API
  - Example Usage of Transcription Endpoint

Groq API is designed to provide fast speech-to-text solution available, offering OpenAI-compatible endpoints that
enable near-instant transcriptions and translations. With Groq API, you can integrate high-quality audio 
processing into your applications at speeds that rival human interaction.

We support two endpoints:

| Endpoint       | Usage                          | API Endpoint                                                |
|----------------|--------------------------------|-------------------------------------------------------------|
| Transcriptions | Convert audio to text          | `https://api.groq.com/openai/v1/audio/transcriptions`        |
| Translations   | Translate audio to English text| `https://api.groq.com/openai/v1/audio/translations`          |

| Model ID                    | Model                | Supported Language(s)          | Description                                                                                                                   |
|-----------------------------|----------------------|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `whisper-large-v3-turbo`    | [Whisper Large V3 Turbo](/docs/model/whisper-large-v3-turbo) | Multilingual                | A fine-tuned version of a pruned Whisper Large V3 designed for fast, multilingual transcription tasks. |
| `whisper-large-v3`          | [Whisper Large V3](/docs/model/whisper-large-v3)     | Multilingual                  | Provides state-of-the-art performance with high accuracy for multilingual transcription and translation tasks. |

## Which Whisper Model Should You Use?
Having more choices is great, but let's try to avoid decision paralysis by breaking down the tradeoffs between models to find the one most suitable for
your applications: 
- If your application is error-sensitive and requires multilingual support, use `whisper-large-v3`. 
- If your application requires multilingual support and you need the best price for performance, use `whisper-large-v3-turbo`.

The following table breaks down the metrics for each model.
| Model | Cost Per Hour | Language Support | Transcription Support | Translation Support | Real-time Speed Factor | Word Error Rate |
|--------|--------|--------|--------|--------|--------|--------|
| `whisper-large-v3` | $0.111 | Multilingual | Yes | Yes | 189 | 10.3% |
| `whisper-large-v3-turbo` | $0.04 | Multilingual | Yes | No | 216 | 12% |

## Working with Audio Files

### Audio File Limitations

* Max File Size: 25 MB (free tier), 100MB (dev tier)
* Max Attachment File Size: 25 MB. If you need to process larger files, use the `url` parameter to specify a url to the file instead.
* Minimum File Length: 0.01 seconds
* Minimum Billed Length: 10 seconds. If you submit a request less than this, you will still be billed for 10 seconds.
* Supported File Types: Either a URL or a direct file upload for `flac`, `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `ogg`, `wav`, `webm`
* Single Audio Track: Only the first track will be transcribed for files with multiple audio tracks. (e.g. dubbed video)
* Supported Response Formats: `json`, `verbose_json`, `text`
* Supported Timestamp Granularities: `segment`, `word`

### Audio Preprocessing
Our speech-to-text models will downsample audio to 16KHz mono before transcribing, which is optimal for speech recognition. This preprocessing can be performed client-side if your original file is extremely 
large and you want to make it smaller without a loss in quality (without chunking, Groq API speech-to-text endpoints accept up to 25MB for free tier and 100MB for [dev tier](/settings/billing)). For lower latency, convert your files to `wav` format. When reducing file size, we recommend FLAC for lossless compression.

The following `ffmpeg` command can be used to reduce file size:

### Working with Larger Audio Files
For audio files that exceed our size limits or require more precise control over transcription, we recommend implementing audio chunking. This process involves:
1. Breaking the audio into smaller, overlapping segments
2. Processing each segment independently
3. Combining the results while handling overlapping

[To learn more about this process and get code for your own implementation, see the complete audio chunking tutorial in our Groq API Cookbook.](https://github.com/groq/groq-api-cookbook/tree/main/tutorials/audio-chunking)

## Using the API 
The following are request parameters you can use in your transcription and translation requests:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | `string` | Required unless using `url` instead | The audio file object for direct upload to translate/transcribe. |
| `url` | `string` | Required unless using `file` instead | The audio URL to translate/transcribe (supports Base64URL). |
| `language` | `string` | Optional | The language of the input audio. Supplying the input language in ISO-639-1 (i.e. `en, `tr`) format will improve accuracy and latency.<br/><br/>The translations endpoint only supports 'en' as a parameter option. |
| `model` | `string` | Required | ID of the model to use.|
| `prompt` | `string` | Optional | Prompt to guide the model's style or specify how to spell unfamiliar words. (limited to 224 tokens) |
| `response_format` | `string` | json | Define the output response format.<br/><br/>Set to `verbose_json` to receive timestamps for audio segments.<br/><br/>Set to `text` to return a text response. |
| `temperature` | `float` | 0 | The temperature between 0 and 1. For translations and transcriptions, we recommend the default value of 0. |
| `timestamp_granularities[]` | `array` | segment | The timestamp granularities to populate for this transcription. `response_format` must be set `verbose_json` to use timestamp granularities.<br/><br/>Either or both of `word` and `segment` are supported. <br/><br/>`segment` returns full metadata and `word` returns only word, start, and end timestamps. To get both word-level timestamps and full segment metadata, include both values in the array. |

### Example Usage of Transcription Endpoint 
The transcription endpoint allows you to transcribe spoken words in audio or video files.

The Groq SDK package can be installed using the following command:

**Examples:**

Example 1 (shell):
```shell
ffmpeg \
  -i <your file> \
  -ar 16000 \
  -ac 1 \
  -map 0:a \
  -c:a flac \
  <output file name>.flac
```

---
