#!/usr/bin/env node
/**
 * Process and diagnose a single issue
 */

const fs = require('fs');

const TEST_PLANS = {
    system: {
        name: 'System Issue Test Plan',
        steps: ['1. Review system logs', '2. Check environment config', '3. Verify dependencies', '4. Test in isolation', '5. Compare with working config'],
        investigation: ['Check application logs', 'Verify environment variables', 'Review recent changes', 'Test minimal configuration'],
        fixes: ['Update dependencies', 'Fix config values', 'Add error handling', 'Increase timeouts'],
        // 優化3: 程式碼修復範例
        codeExamples: {
            errorHandling: `// 增加錯誤處理
try {
    const result = await riskyOperation();
    return result;
} catch (error) {
    console.error('Operation failed:', error);
    // 優雅降級或重試
    return fallbackValue;
}`,
            configFix: `// 修正配置載入
const config = {
    timeout: process.env.TIMEOUT || 5000,
    retries: process.env.RETRIES || 3,
    // 確保必要配置存在
    apiUrl: process.env.API_URL || (() => { throw new Error('API_URL is required'); })()
};`,
            dependencyCheck: `// 檢查依賴版本
const pkg = require('./package.json');
const semver = require('semver');
if (!semver.satisfies(process.version, pkg.engines.node)) {
    console.error(\`Node.js version \${pkg.engines.node} required\`);
    process.exit(1);
}`
        }
    },
    ui: {
        name: 'UI Issue Test Plan',
        steps: ['1. Reproduce on affected browser', '2. Check console for JS errors', '3. Inspect element styles', '4. Test responsive breakpoints', '5. Cross-browser test'],
        investigation: ['DevTools Console', 'Network tab', 'CSS computed styles', 'Multiple browsers'],
        fixes: ['Fix event listener', 'Correct CSS', 'Add null checks', 'Fix breakpoints'],
        codeExamples: {
            eventListener: `// 修正事件監聽器
// 錯誤：元素還沒載入就綁定事件
// button.addEventListener('click', handler);

// 正確：等 DOM 載入完成
document.addEventListener('DOMContentLoaded', () => {
    const button = document.getElementById('myButton');
    if (button) {
        button.addEventListener('click', handler);
    }
});`,
            nullCheck: `// 增加空值檢查
// 錯誤
const value = data.user.profile.name;

// 正確
const value = data?.user?.profile?.name ?? 'Default Name';`,
            cssResponsive: `/* 修正響應式問題 */
.container {
    width: 100%;
    max-width: 1200px;
    padding: 0 15px;
}

@media (max-width: 768px) {
    .container {
        padding: 0 10px;
    }
    .sidebar {
        display: none;
    }
}`
        }
    },
    account: {
        name: 'Account Issue Test Plan',
        steps: ['1. Verify auth flow', '2. Check token validity', '3. Test permissions', '4. Review auth logs', '5. Test user roles'],
        investigation: ['Token expiration', 'User permissions', 'Session storage', 'OAuth flow'],
        fixes: ['Refresh tokens', 'Fix permission checks', 'Correct sessions', 'Update middleware'],
        codeExamples: {
            tokenRefresh: `// 自動刷新 Token
async function fetchWithAuth(url, options = {}) {
    let token = getAccessToken();
    
    if (isTokenExpired(token)) {
        token = await refreshAccessToken();
    }
    
    return fetch(url, {
        ...options,
        headers: {
            ...options.headers,
            'Authorization': \`Bearer \${token}\`
        }
    });
}`,
            permissionCheck: `// 權限檢查中間件
function requirePermission(permission) {
    return (req, res, next) => {
        if (!req.user) {
            return res.status(401).json({ error: 'Unauthorized' });
        }
        if (!req.user.permissions.includes(permission)) {
            return res.status(403).json({ error: 'Forbidden' });
        }
        next();
    };
}

// 使用
app.delete('/api/users/:id', requirePermission('admin'), deleteUser);`
        }
    },
    data: {
        name: 'Data Issue Test Plan',
        steps: ['1. Verify data integrity', '2. Check sync status', '3. Test CRUD', '4. Review queries', '5. Validate transforms'],
        investigation: ['Query for corruption', 'Check timestamps', 'Review validation', 'Test sample data'],
        fixes: ['Add validation', 'Fix sync logic', 'Correct queries', 'Add transactions'],
        codeExamples: {
            validation: `// 資料驗證
const Joi = require('joi');

const userSchema = Joi.object({
    name: Joi.string().min(2).max(100).required(),
    email: Joi.string().email().required(),
    age: Joi.number().integer().min(0).max(150)
});

function validateUser(data) {
    const { error, value } = userSchema.validate(data);
    if (error) {
        throw new Error(\`Validation failed: \${error.message}\`);
    }
    return value;
}`,
            transaction: `// 使用交易確保資料一致性
async function transferFunds(fromId, toId, amount) {
    const session = await mongoose.startSession();
    session.startTransaction();
    
    try {
        await Account.updateOne({ _id: fromId }, { $inc: { balance: -amount } }, { session });
        await Account.updateOne({ _id: toId }, { $inc: { balance: amount } }, { session });
        await session.commitTransaction();
    } catch (error) {
        await session.abortTransaction();
        throw error;
    } finally {
        session.endSession();
    }
}`
        }
    },
    performance: {
        name: 'Performance Issue Test Plan',
        steps: ['1. Baseline performance', '2. Profile slow ops', '3. Check N+1 queries', '4. Monitor memory', '5. Find bottlenecks'],
        investigation: ['Profile functions', 'Query performance', 'Network requests', 'Caching strategy'],
        fixes: ['Add caching', 'Batch queries', 'Lazy loading', 'Optimize algorithms'],
        codeExamples: {
            caching: `// 增加快取
const cache = new Map();
const CACHE_TTL = 5 * 60 * 1000; // 5 分鐘

async function getCachedData(key, fetchFn) {
    const cached = cache.get(key);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
        return cached.data;
    }
    
    const data = await fetchFn();
    cache.set(key, { data, timestamp: Date.now() });
    return data;
}`,
            batchQuery: `// 批次查詢避免 N+1
// 錯誤：N+1 查詢
for (const user of users) {
    const posts = await db.query('SELECT * FROM posts WHERE user_id = ?', [user.id]);
}

// 正確：批次查詢
const userIds = users.map(u => u.id);
const allPosts = await db.query(
    'SELECT * FROM posts WHERE user_id IN (?)',
    [userIds]
);
const postsByUser = groupBy(allPosts, 'user_id');`,
            lazyLoad: `// React 懶加載
const HeavyComponent = React.lazy(() => import('./HeavyComponent'));

function App() {
    return (
        <Suspense fallback={<Loading />}>
            <HeavyComponent />
        </Suspense>
    );
}`
        }
    },
    feature: {
        name: 'Feature Request Analysis',
        steps: ['1. Review requirements', '2. Identify components', '3. Design approach', '4. Estimate effort', '5. Plan testing'],
        investigation: ['Clarify requirements', 'Check similar features', 'Architecture impact', 'Dependencies'],
        fixes: ['New component', 'Extend existing', 'Add config option', 'Create API endpoint'],
        codeExamples: {}
    }
};

const KEYWORDS = {
    error: ['error', 'exception', 'fail', 'crash', '錯誤', '失敗'],
    performance: ['slow', 'lag', 'delay', 'timeout', '慢', '延遲'],
    ui: ['button', 'display', 'click', 'layout', '顯示', '按鈕'],
    data: ['data', 'sync', 'save', 'load', '資料', '同步'],
    auth: ['login', 'auth', 'permission', '登入', '權限']
};

function parseArgs() {
    const args = { file: null, index: 0, output: null, json: false };
    const argv = process.argv.slice(2);
    for (let i = 0; i < argv.length; i++) {
        switch (argv[i]) {
            case '--file': args.file = argv[++i]; break;
            case '--index': args.index = parseInt(argv[++i]); break;
            case '--output': args.output = argv[++i]; break;
            case '--json': args.json = true; break;
            case '--help': console.log('Usage: node process-issue.js [--file FILE] [--output FILE] [--json]'); process.exit(0);
        }
    }
    return args;
}

function analyzeIssue(issue) {
    const combined = `${issue.title || ''} ${issue.description || ''}`.toLowerCase();
    const detected = {};
    Object.entries(KEYWORDS).forEach(([cat, words]) => { detected[cat] = words.some(w => combined.includes(w)); });
    
    const suggestions = [];
    if (detected.error) suggestions.push('Check error logs and stack traces');
    if (detected.performance) suggestions.push('Profile performance');
    if (detected.ui) suggestions.push('Inspect DOM and CSS');
    if (detected.data) suggestions.push('Verify data flow');
    if (detected.auth) suggestions.push('Review auth handling');
    if (suggestions.length === 0) suggestions.push('Review related code', 'Check recent changes');
    
    return { issueId: issue.id, title: issue.title, group: issue.group, priority: issue.priority,
             affectedAreas: Object.keys(detected).filter(k => detected[k]), suggestions };
}

function generateTestPlan(issue) {
    const template = TEST_PLANS[issue.group] || TEST_PLANS.system;
    return { 
        issueId: issue.id, 
        planName: template.name, 
        steps: template.steps,
        investigation: template.investigation, 
        fixes: template.fixes,
        codeExamples: template.codeExamples || {}
    };
}

function generateReport(issue, analysis, plan) {
    const sep = '='.repeat(60);
    const sep2 = '-'.repeat(40);
    let r = `${sep}\nISSUE DIAGNOSIS REPORT\n${sep}\n\n`;
    r += `Issue ID: ${issue.id}\nTitle: ${issue.title}\nCategory: ${issue.group || 'unknown'}\n`;
    r += `Priority: ${(issue.priority || 'medium').toUpperCase()}\nStatus: ${issue.status || 'open'}\n`;
    r += `Reporter: ${issue.reporter || 'Anonymous'}\nDevice: ${issue.device || 'Unknown'}/${issue.browser || 'Unknown'}\n\n`;
    
    r += `${sep2}\nDESCRIPTION:\n${sep2}\n${issue.description || 'No description'}\n\n`;
    
    // 如果有重現步驟
    if (issue.stepsToReproduce) {
        r += `${sep2}\nSTEPS TO REPRODUCE:\n${sep2}\n${issue.stepsToReproduce}\n\n`;
    }
    
    // 如果有錯誤訊息
    if (issue.errorMessage) {
        r += `${sep2}\nERROR MESSAGE:\n${sep2}\n${issue.errorMessage}\n\n`;
    }
    
    r += `${sep2}\nANALYSIS:\n${sep2}\nAffected Areas: ${analysis.affectedAreas.join(', ') || 'General'}\n\nSuggested Investigations:\n`;
    analysis.suggestions.forEach((s, i) => { r += `  ${i + 1}. ${s}\n`; });
    
    r += `\n${sep2}\nTEST PLAN: ${plan.planName}\n${sep2}\n\nSteps:\n`;
    plan.steps.forEach(s => { r += `  ${s}\n`; });
    r += '\nInvestigation Focus:\n';
    plan.investigation.forEach(s => { r += `  • ${s}\n`; });
    r += '\nCommon Fixes:\n';
    plan.fixes.forEach(s => { r += `  • ${s}\n`; });
    
    // 優化3: 加入程式碼範例
    if (plan.codeExamples && Object.keys(plan.codeExamples).length > 0) {
        r += `\n${sep2}\nCODE FIX EXAMPLES:\n${sep2}\n`;
        Object.entries(plan.codeExamples).forEach(([name, code]) => {
            r += `\n【${name}】\n\`\`\`\n${code}\n\`\`\`\n`;
        });
    }
    
    r += `\n${sep2}\nACTION ITEMS:\n${sep2}\n[ ] Reproduce issue\n[ ] Identify root cause\n[ ] Implement fix\n[ ] Write tests\n[ ] Verify fix\n[ ] Update status\n\n`;
    r += `${sep2}\nROOT CAUSE:\n${sep2}\n(Document after investigation)\n\n`;
    r += `${sep2}\nFIX IMPLEMENTED:\n${sep2}\n(Document the fix)\n\n`;
    r += `${sep}\nGenerated: ${new Date().toLocaleString('zh-TW')}\n${sep}\n`;
    return r;
}

function processIssue(issue) {
    console.error(`\nProcessing: ${issue.title} [${(issue.priority || 'medium').toUpperCase()}]`);
    const analysis = analyzeIssue(issue);
    console.error(`  Areas: ${analysis.affectedAreas.join(', ') || 'General'}`);
    const plan = generateTestPlan(issue);
    console.error(`  Plan: ${plan.planName}`);
    const report = generateReport(issue, analysis, plan);
    return { issue, analysis, plan, report, status: 'diagnosed', processedAt: new Date().toISOString() };
}

async function readStdin() {
    return new Promise((resolve) => {
        if (process.stdin.isTTY) { resolve(null); return; }
        let data = '';
        process.stdin.setEncoding('utf8');
        process.stdin.on('data', chunk => { data += chunk; });
        process.stdin.on('end', () => { try { resolve(JSON.parse(data)); } catch { resolve(null); } });
    });
}

async function main() {
    const args = parseArgs();
    let issue = null;
    
    if (args.file) {
        const data = JSON.parse(fs.readFileSync(args.file, 'utf8'));
        issue = Array.isArray(data) ? data[args.index] : (data.issues ? data.issues[args.index] : data);
    } else {
        const data = await readStdin();
        if (data) issue = Array.isArray(data) ? data[args.index] : (data.issues ? data.issues[args.index] : data);
    }
    
    if (!issue) { console.error('No issue data. Use --file or pipe JSON.'); process.exit(1); }
    
    const result = processIssue(issue);
    const output = args.json 
        ? JSON.stringify({ issueId: result.issue.id, status: result.status, areas: result.analysis.affectedAreas, plan: result.plan.planName }, null, 2)
        : result.report;
    
    if (args.output) { fs.writeFileSync(args.output, output, 'utf8'); console.error(`Saved to: ${args.output}`); }
    else console.log(output);
}

module.exports = { processIssue, analyzeIssue, generateTestPlan, TEST_PLANS };
if (require.main === module) main().catch(e => { console.error(e.message); process.exit(1); });
