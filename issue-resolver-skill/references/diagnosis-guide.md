# Issue Diagnosis Reference Guide

## Quick Diagnosis by Symptom

### Error Messages
| Symptom | Check First | Common Causes |
|---------|-------------|---------------|
| "Cannot read property X of undefined" | Null checks, async timing | Data not loaded, wrong API response |
| "Network error" | API endpoint, CORS | Server down, wrong URL, CORS policy |
| "Permission denied" | Auth token, user role | Expired token, insufficient permissions |
| "Out of memory" | Memory leaks, large data | Unbounded arrays, event listener leaks |
| "Timeout" | API response time, loops | Slow queries, infinite loops |

### UI Issues
| Symptom | Check First | Common Causes |
|---------|-------------|---------------|
| Button not responding | Event listeners, z-index | Listener not attached, element covered |
| Layout broken | CSS, viewport | Missing styles, responsive breakpoints |
| Content not showing | Data binding, display | API failure, display:none, v-if/ng-if |
| Flickering | Re-renders, transitions | Frequent state updates, CSS transitions |

### Performance Issues
| Symptom | Check First | Common Causes |
|---------|-------------|---------------|
| Slow page load | Network, bundle size | Large assets, too many requests |
| Slow interaction | JS execution, re-renders | Expensive operations in handlers |
| Memory growth | Listeners, closures | Event listeners not removed |
| High CPU | Loops, animations | Infinite loops, heavy animations |

## Investigation Commands

### Browser DevTools
```javascript
// Check for errors
// Console tab

// Monitor network
// Network tab

// Profile performance  
// Performance tab > Record

// Memory analysis
// Memory tab > Heap snapshot

// Check event listeners
getEventListeners(document.querySelector('#myButton'))
```

### Node.js
```bash
# Check logs
tail -f app.log

# Monitor process
top -p $(pgrep -f "node")

# Debug mode
NODE_DEBUG=* node app.js

# Profile
node --prof app.js
```

### Git Investigation
```bash
# Recent changes
git log --oneline -20

# Changes to file
git log -p -- path/to/file

# Who changed what
git blame path/to/file

# Find when bug introduced
git bisect start
git bisect bad HEAD
git bisect good v1.0.0
```

## Common Fix Patterns

### Null/Undefined Errors
```javascript
// Before
const value = obj.prop.nested;

// After
const value = obj?.prop?.nested ?? defaultValue;
```

### Race Conditions
```javascript
// Before
fetchData();
processData(data); // data not ready

// After
const data = await fetchData();
processData(data);
```

### Memory Leaks
```javascript
// Before
useEffect(() => {
  const interval = setInterval(update, 1000);
}, []);

// After
useEffect(() => {
  const interval = setInterval(update, 1000);
  return () => clearInterval(interval);
}, []);
```

### N+1 Queries
```javascript
// Before
for (const user of users) {
  const posts = await db.query('SELECT * FROM posts WHERE user_id = ?', [user.id]);
}

// After
const userIds = users.map(u => u.id);
const allPosts = await db.query('SELECT * FROM posts WHERE user_id IN (?)', [userIds]);
```

## Verification Checklist

After implementing a fix:

- [ ] Issue reproducible before fix
- [ ] Issue resolved after fix
- [ ] No new errors in console
- [ ] Related functionality works
- [ ] Edge cases handled
- [ ] Tests pass
- [ ] Performance not degraded
