---
name: code-reviewer
description: |
  Review code for quality, security, performance, and best practices. Identifies bugs, security vulnerabilities, and improvement opportunities.
  Use when: reviewing pull requests, auditing code quality, performing security reviews, checking code before deployment, or when user mentions code review, PR review, 程式碼審查, 審查, review, 檢查程式碼.
  Triggers: "review code", "code review", "PR review", "check my code", "審查", "review this", "程式碼檢查", "看一下這段程式"
allowed-tools: Read, Grep, Glob
version: 1.0.0
---

# Code Reviewer

Perform comprehensive code reviews focusing on quality, security, and maintainability.

## Review Dimensions

| Dimension | Focus Areas |
|-----------|-------------|
| **Correctness** | Logic errors, edge cases, null handling |
| **Security** | OWASP Top 10, injection, auth flaws |
| **Performance** | Complexity, memory, database queries |
| **Maintainability** | Readability, naming, documentation |
| **Testing** | Coverage, edge cases, mocking |

## Review Checklist

### 1. Correctness

```markdown
- [ ] Logic matches requirements
- [ ] Edge cases handled (empty, null, boundary)
- [ ] Error handling is appropriate
- [ ] Race conditions considered
- [ ] State mutations are intentional
- [ ] Return values are correct
```

### 2. Security (OWASP Top 10)

```markdown
- [ ] No SQL/NoSQL injection vulnerabilities
- [ ] No XSS vulnerabilities
- [ ] No command injection
- [ ] No path traversal
- [ ] Authentication properly implemented
- [ ] Authorization checks present
- [ ] Sensitive data not logged
- [ ] Secrets not hardcoded
- [ ] CSRF protection in place
- [ ] Input validation on all user data
```

#### Common Vulnerabilities

```python
# BAD: SQL Injection
query = f"SELECT * FROM users WHERE id = {user_id}"

# GOOD: Parameterized query
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))
```

```javascript
// BAD: XSS vulnerability
element.innerHTML = userInput;

// GOOD: Safe text content
element.textContent = userInput;
// Or sanitize HTML
element.innerHTML = DOMPurify.sanitize(userInput);
```

```python
# BAD: Command injection
os.system(f"convert {user_file} output.png")

# GOOD: Use subprocess with list
subprocess.run(["convert", user_file, "output.png"], check=True)
```

### 3. Performance

```markdown
- [ ] No N+1 query problems
- [ ] Appropriate indexing used
- [ ] No unnecessary loops/iterations
- [ ] Caching considered for expensive operations
- [ ] Pagination for large datasets
- [ ] No memory leaks
- [ ] Async operations where beneficial
```

#### Common Issues

```python
# BAD: N+1 queries
for user in users:
    orders = db.query(Order).filter(Order.user_id == user.id).all()

# GOOD: Eager loading
users = db.query(User).options(joinedload(User.orders)).all()
```

```javascript
// BAD: Creating objects in loop
items.map(item => {
  const config = { ...defaultConfig }; // New object each iteration
  return process(item, config);
});

// GOOD: Reuse when possible
const config = { ...defaultConfig };
items.map(item => process(item, config));
```

### 4. Maintainability

```markdown
- [ ] Functions are small and focused
- [ ] Naming is clear and consistent
- [ ] No magic numbers/strings
- [ ] DRY principle followed
- [ ] Comments explain "why" not "what"
- [ ] Code is self-documenting
- [ ] Consistent formatting
- [ ] No dead code
```

#### Naming Guidelines

```python
# BAD: Unclear names
def proc(d):
    return d['x'] * 2

# GOOD: Descriptive names
def calculate_double_quantity(order_data: dict) -> int:
    return order_data['quantity'] * 2
```

### 5. Testing

```markdown
- [ ] Unit tests for new logic
- [ ] Edge cases covered
- [ ] Error paths tested
- [ ] Mocks used appropriately
- [ ] Tests are deterministic
- [ ] No test interdependencies
```

## Review Comment Templates

### Bug Found

```markdown
🐛 **Bug**: [Brief description]

**Problem**: [Explanation of the issue]

**Impact**: [What could go wrong]

**Suggested fix**:
```code
[Corrected code]
```
```

### Security Issue

```markdown
🔒 **Security**: [Vulnerability type]

**Risk**: [Severity: Critical/High/Medium/Low]

**Problem**: [How it can be exploited]

**Fix**:
```code
[Secure implementation]
```

**Reference**: [OWASP link or documentation]
```

### Performance Concern

```markdown
⚡ **Performance**: [Issue type]

**Impact**: [Time/memory/scaling concern]

**Current complexity**: O(n²)
**Suggested complexity**: O(n)

**Improvement**:
```code
[Optimized code]
```
```

### Suggestion

```markdown
💡 **Suggestion**: [Brief description]

**Rationale**: [Why this would be better]

**Example**:
```code
[Alternative implementation]
```
```

### Question

```markdown
❓ **Question**: [Your question]

**Context**: [Why you're asking]
```

## Review Output Format

```markdown
# Code Review: [PR Title / File Name]

## Summary
[2-3 sentence overview of the changes and overall assessment]

## Critical Issues 🚨
[Must fix before merge]

## Security Concerns 🔒
[Security-related findings]

## Performance ⚡
[Performance observations]

## Code Quality 📝
[Maintainability, style, best practices]

## Suggestions 💡
[Nice-to-have improvements]

## Questions ❓
[Clarifications needed]

## Verdict
- [ ] ✅ Approve
- [ ] 🔄 Request changes
- [ ] 💬 Comment only
```

## Language-Specific Checks

### Python
- Type hints present
- Docstrings for public functions
- No mutable default arguments
- Context managers for resources
- f-strings over .format()

### JavaScript/TypeScript
- TypeScript types defined
- No `any` type abuse
- Proper async/await usage
- No callback hell
- ESLint rules followed

### React
- Keys in list rendering
- useCallback/useMemo where needed
- No prop drilling (use context)
- Proper cleanup in useEffect
- Accessible components

### SQL
- Indexes for filtered columns
- No SELECT *
- Proper JOIN types
- Avoid LIKE '%pattern%'
- Transactions for multi-step operations

## Severity Levels

| Level | Description | Action |
|-------|-------------|--------|
| 🚨 Critical | Security vulnerability, data loss risk | Block merge |
| 🔴 High | Bugs, major issues | Must fix |
| 🟡 Medium | Code smell, minor issues | Should fix |
| 🟢 Low | Style, suggestions | Nice to have |
| 💬 Info | Questions, observations | No action required |
