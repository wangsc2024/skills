---
name: internal-comms
description: |
  Draft professional internal communications including status reports, team updates, incident reports, and company newsletters. Provides structured templates and tone guidance.
  Use when: writing team updates, status reports, incident reports, FAQs, newsletters, or when user mentions 報告, status report, 進度報告, 週報, newsletter, incident report, 事件報告, 內部溝通.
  Triggers: "status report", "報告", "進度報告", "週報", "newsletter", "incident report", "事件報告", "內部溝通", "3P更新"
---

# Internal Communications

Draft professional internal communications using structured templates.

## Communication Types

| Type | Purpose | Frequency |
|------|---------|-----------|
| **3P Update** | Progress, Plans, Problems | Weekly |
| **Status Report** | Project/team status | Weekly/Monthly |
| **Incident Report** | Issue documentation | As needed |
| **Company Newsletter** | Org-wide updates | Monthly |
| **Leadership Update** | Executive summary | Weekly |
| **FAQ Response** | Answer common questions | As needed |

## 3P Update (Progress, Plans, Problems)

### Template

```markdown
# 3P Update: [Team/Project Name]
**Week of**: [Date]
**Author**: [Name]

## Progress (What we accomplished)
- [Completed item 1]
- [Completed item 2]
- [Completed item 3]

## Plans (What we're working on next)
- [Planned item 1] - ETA: [Date]
- [Planned item 2] - ETA: [Date]
- [Planned item 3] - ETA: [Date]

## Problems (Blockers and risks)
- **[Issue 1]**: [Description] - Need: [What help is needed]
- **[Issue 2]**: [Description] - Status: [Being addressed / Needs escalation]

## Metrics
| Metric | Last Week | This Week | Target |
|--------|-----------|-----------|--------|
| [KPI 1] | X | Y | Z |

## Highlights
[One notable achievement or learning from this week]
```

### Example

```markdown
# 3P Update: Platform Team
**Week of**: January 15, 2024
**Author**: Jane Smith

## Progress
- Deployed v2.3 authentication service to production
- Reduced API latency by 15% through caching optimization
- Completed security audit remediation (12/12 items)

## Plans
- Migrate user database to new cluster - ETA: Jan 22
- Implement rate limiting for public API - ETA: Jan 25
- Begin Q1 capacity planning - ETA: Jan 26

## Problems
- **Redis cluster**: Experiencing intermittent connection timeouts
  - Need: DevOps review of network configuration
- **Vendor delay**: Third-party auth provider delayed SDK update
  - Status: Workaround in place, monitoring

## Metrics
| Metric | Last Week | This Week | Target |
|--------|-----------|-----------|--------|
| Uptime | 99.92% | 99.98% | 99.95% |
| P95 Latency | 245ms | 208ms | <200ms |
| Error Rate | 0.12% | 0.08% | <0.1% |

## Highlights
Team successfully handled 3x normal traffic during product launch with zero incidents.
```

## Status Report

### Template

```markdown
# [Project/Team] Status Report
**Period**: [Start Date] - [End Date]
**Status**: 🟢 On Track / 🟡 At Risk / 🔴 Blocked

## Executive Summary
[2-3 sentences summarizing overall status and key points]

## Accomplishments
1. [Major accomplishment with impact]
2. [Major accomplishment with impact]
3. [Major accomplishment with impact]

## Upcoming Milestones
| Milestone | Target Date | Status | Owner |
|-----------|-------------|--------|-------|
| [Milestone 1] | [Date] | [Status] | [Name] |

## Risks & Issues
| Risk/Issue | Impact | Mitigation | Owner |
|------------|--------|------------|-------|
| [Risk 1] | High/Med/Low | [Action] | [Name] |

## Resource Needs
- [Resource need 1]
- [Resource need 2]

## Key Decisions Needed
- [ ] [Decision 1] - Needed by: [Date]
- [ ] [Decision 2] - Needed by: [Date]
```

## Incident Report

### Template

```markdown
# Incident Report: [Incident Title]
**Incident ID**: INC-[Number]
**Severity**: P1/P2/P3/P4
**Status**: Resolved / Investigating / Monitoring

## Summary
**Duration**: [Start Time] - [End Time] ([X hours/minutes])
**Impact**: [What was affected and how many users]
**Root Cause**: [Brief description]

## Timeline
| Time (UTC) | Event |
|------------|-------|
| HH:MM | [Event 1] |
| HH:MM | [Event 2] |
| HH:MM | [Event 3] |

## Impact Assessment
- **Users Affected**: [Number or percentage]
- **Services Affected**: [List of services]
- **Revenue Impact**: [If applicable]
- **Data Loss**: [Yes/No - details if yes]

## Root Cause Analysis
[Detailed explanation of what caused the incident]

## Resolution
[What was done to resolve the incident]

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| [Action 1] | [Name] | [Date] | Open |

## Lessons Learned
1. [What we learned]
2. [What we'll do differently]

## Prevention Measures
- [Measure 1]
- [Measure 2]
```

## Company Newsletter

### Template

```markdown
# [Company] Newsletter
**[Month Year]**

---

## From Leadership
[Brief message from CEO/leadership - 2-3 paragraphs]

---

## Company Updates

### 📈 Business Highlights
- [Highlight 1]
- [Highlight 2]
- [Highlight 3]

### 🚀 Product Updates
- **[Product/Feature]**: [Brief description]
- **[Product/Feature]**: [Brief description]

### 👥 Team News
- Welcome [New Hires] joining [Teams]
- Congratulations to [Name] on [Achievement]
- [Team] celebrating [Milestone]

---

## Upcoming Events
| Date | Event | Details |
|------|-------|---------|
| [Date] | [Event] | [Brief description] |

---

## Spotlight: [Team/Individual]
[2-3 paragraph feature on a team or individual achievement]

---

## Resources & Reminders
- [Important deadline or reminder]
- [New resource or policy update]

---

*Questions? Reach out to [Contact]*
```

## Leadership Update

### Template

```markdown
# Leadership Update
**Week of [Date]**

## Key Messages
1. [Most important message]
2. [Second priority message]
3. [Third priority message]

## Strategic Updates
### [Initiative 1]
[Brief status and next steps]

### [Initiative 2]
[Brief status and next steps]

## Metrics Dashboard
| Area | Status | Trend | Notes |
|------|--------|-------|-------|
| Revenue | 🟢 | ↑ | [Context] |
| Operations | 🟡 | → | [Context] |
| Customer | 🟢 | ↑ | [Context] |

## Decisions Made
- [Decision 1]: [Rationale]
- [Decision 2]: [Rationale]

## Looking Ahead
[Key focus areas for next week]

## Ask of Leaders
[Specific actions needed from leadership team]
```

## FAQ Response

### Template

```markdown
# FAQ: [Topic]

## Q: [Question 1]
**A**: [Clear, concise answer]

[Additional context if needed]

---

## Q: [Question 2]
**A**: [Clear, concise answer]

**Related resources**:
- [Link to resource 1]
- [Link to resource 2]

---

## Q: [Question 3]
**A**: [Clear, concise answer]

> **Note**: [Important caveat or clarification]

---

## Still have questions?
Contact [Team/Person] at [Email/Channel]
```

## Writing Guidelines

### Tone

| Context | Tone | Example |
|---------|------|---------|
| Good news | Celebratory but professional | "We're excited to announce..." |
| Bad news | Direct, empathetic | "We want to be transparent about..." |
| Neutral update | Clear, informative | "Here's this week's update..." |
| Urgent issue | Calm, action-oriented | "We've identified an issue and are actively working to resolve it." |

### Best Practices

```markdown
✅ DO:
- Lead with the most important information
- Use bullet points for scanability
- Include specific dates and owners
- Provide context for metrics
- End with clear next steps or calls to action

❌ DON'T:
- Bury bad news at the end
- Use jargon without explanation
- Leave ambiguity about ownership
- Forget to proofread
- Send without appropriate approvals
```

### Formatting Tips

```markdown
- Use **bold** for emphasis on key points
- Use tables for structured data
- Use emoji sparingly for visual breaks: 🟢🟡🔴📈🚀
- Keep paragraphs short (3-4 sentences max)
- Include TL;DR for long communications
```

## Checklist

Before sending:

- [ ] Clear subject line
- [ ] Appropriate recipients (To/CC)
- [ ] Key message in first paragraph
- [ ] Specific dates and owners for action items
- [ ] Proofread for errors
- [ ] Approved by stakeholders (if required)
- [ ] Mobile-friendly formatting
- [ ] Links tested
