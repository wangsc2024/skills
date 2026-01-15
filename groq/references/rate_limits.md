# Groq - Rate Limits

**Pages:** 2

---

## Rate Limits

**URL:** llms-txt#rate-limits

**Contents:**
- Understanding Rate Limits
- Rate Limits
- Rate Limit Headers
- Handling Rate Limits
- Need Higher Rate Limits?
- Wolfram Alpha: Quickstart (js)
- Print the final content

Rate limits act as control measures to regulate how frequently users and applications can access our API within specified timeframes. These limits help ensure service stability, fair access, and protection
against misuse so that we can serve reliable and fast inference for all.

## Understanding Rate Limits
Rate limits are measured in:
- **RPM:** Requests per minute
- **RPD:** Requests per day
- **TPM:** Tokens per minute
- **TPD:** Tokens per day
- **ASH:** Audio seconds per hour
- **ASD:** Audio seconds per day

Cached tokens do not count towards your rate limits.

Rate limits apply at the organization level, not individual users. You can hit any limit type depending on which threshold you reach first.

**Example:** Let's say your RPM = 50 and your TPM = 200K. If you were to send 50 requests with only 100 tokens within a minute, you would reach your limit even though you did not send 200K tokens within those
50 requests.

## Rate Limits
The following is a high level summary and there may be  exceptions to these limits. You can view the current, exact rate limits for your organization on the [limits page](/settings/limits) in your account settings.

## Rate Limit Headers
In addition to viewing your limits on your account's [limits](https://console.groq.com/settings/limits) page, you can also view rate limit information such as remaining requests and tokens in HTTP response 
headers as follows:

The following headers are set (values are illustrative):

## Handling Rate Limits
When you exceed rate limits, our API returns a `429 Too Many Requests` HTTP status code.

**Note**: `retry-after` is only set if you hit the rate limit and status code 429 is returned. The other headers are always included.

## Need Higher Rate Limits?
If you need higher rate limits, you can [request them here](https://groq.com/self-serve-support).

## Wolfram Alpha: Quickstart (js)

URL: https://console.groq.com/docs/wolfram-alpha/scripts/quickstart

## Print the final content

URL: https://console.groq.com/docs/wolfram-alpha/scripts/quickstart.py

```python
import json
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
            "content": "What is 1293392*29393?",
        }
    ],
    model="groq/compound",
    compound_custom={
        "tools": {
            "enabled_tools": ["wolfram_alpha"],
            "wolfram_settings": {"authorization": "your_wolfram_alpha_api_key_here"}
        }
    }
)

message = chat_completion.choices[0].message

**Examples:**

Example 1 (javascript):
```javascript
import { Groq } from "groq-sdk";

const groq = new Groq({
  defaultHeaders: {
    "Groq-Model-Version": "latest"
  }
});

const chatCompletion = await groq.chat.completions.create({
  messages: [
    {
      role: "user",
      content: "What is 1293392*29393?",
    },
  ],
  model: "groq/compound",
  compound_custom: {
    tools: {
      enabled_tools: ["wolfram_alpha"],
      wolfram_settings: { authorization: "your_wolfram_alpha_api_key_here" }
    }
  }
});

const message = chatCompletion.choices[0].message;

// Print the final content
console.log(message.content);

// Print the reasoning process
console.log(message.reasoning);

// Print the first executed tool
console.log(message.executed_tools[0]);
```

---

## Spend Limits

**URL:** llms-txt#spend-limits

**Contents:**
- Quick Start
- How It Works
- Setting Up Spending Limits
  - Create a Spending Limit
  - Add Usage Alerts
  - Manage Your Alerts
- Email Notifications
- Best Practices
  - Setting Effective Limits
- Troubleshooting

Control your API costs with automated spending limits and proactive usage alerts when approaching budget thresholds.

**Set a spending limit in 3 steps:**
1. Go to [**Settings** → **Billing** → **Limits**](/settings/billing/limits)
2. Click **Add Limit** and enter your monthly budget in USD
3. Add alert thresholds at 50%, 75%, and 90% of your limit
4. Click **Save** to activate the limit

**Requirements:** Paid tier account with organization owner permissions.

Spend limits automatically protect your budget by blocking API access when you reach your monthly cap. The limit applies organization-wide across all API keys, so usage from any team member or application counts toward the same shared limit. If you hit your set limit, API calls from any key in your organization will return a 400 with code `blocked_api_access`. Usage alerts notify you via email before you hit the limit, giving you time to adjust usage or increase your budget. 
<br />
This feature offers:

- **Near real-time tracking:** Your current spend updates every 10-15 minutes  
- **Automatic monthly reset:** Limits reset at the beginning of each billing cycle (1st of the month)  
- **Immediate blocking:** API access is blocked when a spend update detects you've hit your limit

<br/>
> ⚠️ **Important:** There's a 10-15 minute delay in spend tracking. This means you might exceed your limit by a small amount during high usage periods.

## Setting Up Spending Limits

### Create a Spending Limit

Navigate to [**Settings** → **Billing** → **Limits**](/settings/billing/limits) and click **Add Limit**.

Example Monthly Spending Limit: $500

Your API requests will be blocked when you reach $500 in monthly usage. The limit resets automatically on the 1st of each month.

Set up email notifications before you hit your limit:
Alert at $250 (50% of limit)
Alert at $375 (75% of limit)
Alert at $450 (90% of limit)

**To add an alert:**
1. Click **Add Alert** in the Usage Alerts section
2. Enter the USD amount trigger
3. Click **Save**

Alerts appear as visual markers on your spending progress bar on Groq Console Limits page under Billing.

### Manage Your Alerts

- **Edit Limit:** Click the pencil icon next to any alert
- **Delete:** Click the trash icon to remove an alert
- **Multiple alerts:** Add as many thresholds as needed

## Email Notifications

All spending alerts and limit notifications are sent from **support@groq.com** to your billing email addresses.

**Update billing emails:**
1. Go to [**Settings** → **Billing** → **Manage**](/settings/billing)
2. Add or update email addresses
3. Return to the Limits page to confirm the changes

**Pro tip:** Add multiple team members to billing emails so important notifications don't get missed.

### Setting Effective Limits

- **Start conservative:** Set your first limit 20-30% above your expected monthly usage to account for variability.

- **Monitor patterns:** Review your usage for 2-3 months, then adjust limits based on actual consumption patterns.

- **Leave buffer room:** Don't set limits exactly at your expected usage—unexpected spikes can block critical API access.

- **Use multiple thresholds:** Set alerts at 50%, 75%, and 90% of your limit to get progressive warnings.

### Can't Access the Limits Page?

- **Check your account tier:** Spending limits are only available on paid plans, not free tier accounts.

- **Verify permissions:** You need organization owner permissions to manage spending limits.

- **Feature availability:** Contact us via support@groq.com if you're on a paid tier but don't see the spending limits feature.

### Not Receiving Alert Emails?

- **Verify email addresses:** Check that your billing emails are correct in [**Settings** → **Billing** → **Manage**](/settings/billing).

- **Check spam folders:** Billing alerts might be filtered by your email provider.

- **Test notifications:** Set a low-dollar test alert to verify email delivery is working.

### API Access Blocked?

- **Check your spending status:** The [limits page](/settings/billing/limits) shows your current spend against your limit.

- **Increase your limit:** You can raise your spending limit at any time to restore immediate access if you've hit your spend limit. You can also remove it to unblock your API access immediately.

- **Wait for reset:** If you've hit your limit, API access will restore on the 1st of the next month.

**Q: Can I set different limits for different API endpoints or API keys?**  
A: No, spending limits are organization-wide and apply to your total monthly usage across all API endpoints and all API keys in your organization. All team members and applications using your organization's API keys share the same spending limit.

**Q: What happens to in-flight requests when I hit my limit?**  
A: In-flight requests complete normally, but new requests are blocked immediately.

**Q: Can I set weekly or daily spending limits?**  
A: Currently, only monthly limits are supported. Limits reset on the 1st of each month.

**Q: How accurate is the spending tracking?**  
A: Spending is tracked in near real-time with a 10-15 minute delay. The delay prevents brief usage spikes from prematurely triggering limits.

**Q: Can I temporarily disable my spending limit?**  
A: Yes, you can edit or remove your spending limit at any time from the limits page.

Need help? Contact our support team at support@groq.com with details about your configuration and any error messages.

## API Error Codes and Responses

URL: https://console.groq.com/docs/errors

---
