#!/usr/bin/env node
/**
 * Fetch pending issues from Gun.js relay server
 */

const Gun = require('gun');
const fs = require('fs');

const CONFIG = {
    relayUrl: process.env.ISSUE_RELAY_URL || 'https://relay-o.oopdoo.org.ua/gun',
    nodePrefix: process.env.ISSUE_NODE_PREFIX || 'issue-tracker',
    timeout: 5000,
    groups: {
        system: { name: '系統問題', icon: '⚙️' },
        ui: { name: '介面問題', icon: '🎨' },
        account: { name: '帳號問題', icon: '👤' },
        data: { name: '資料問題', icon: '💾' },
        performance: { name: '效能問題', icon: '⚡' },
        feature: { name: '功能建議', icon: '💡' },
        other: { name: '其他', icon: '📋' }
    },
    priorities: {
        critical: { name: '緊急', icon: '🔴', weight: 4 },
        high: { name: '高', icon: '🟠', weight: 3 },
        medium: { name: '中', icon: '🟡', weight: 2 },
        low: { name: '低', icon: '🟢', weight: 1 }
    },
    statuses: {
        open: { name: '待處理', icon: '📬' },
        'in-progress': { name: '處理中', icon: '🔧' },
        resolved: { name: '已解決', icon: '✅' },
        closed: { name: '已關閉', icon: '🔒' }
    }
};

function parseArgs() {
    const args = { relay: CONFIG.relayUrl, prefix: CONFIG.nodePrefix, group: null, priority: null, status: 'open,in-progress', limit: null, format: 'json', output: null, timeout: CONFIG.timeout, system: null };
    const argv = process.argv.slice(2);
    for (let i = 0; i < argv.length; i++) {
        switch (argv[i]) {
            case '--relay': args.relay = argv[++i]; break;
            case '--prefix': args.prefix = argv[++i]; break;
            case '--group': args.group = argv[++i]; break;
            case '--priority': args.priority = argv[++i]; break;
            case '--status': args.status = argv[++i]; break;
            case '--limit': args.limit = parseInt(argv[++i]); break;
            case '--format': args.format = argv[++i]; break;
            case '--output': args.output = argv[++i]; break;
            case '--timeout': args.timeout = parseInt(argv[++i]); break;
            case '--system': args.system = argv[++i]; break;
            case '--help': console.log('Usage: node fetch-issues.js [--relay URL] [--group GROUP] [--priority PRIORITY] [--system SYSTEM] [--format json|markdown|summary] [--output FILE]'); process.exit(0);
        }
    }
    return args;
}

async function fetchIssues(args) {
    return new Promise((resolve) => {
        const gun = Gun({ peers: [args.relay], localStorage: false, radisk: false });
        const issues = [];
        const seen = new Set();
        const statusFilter = args.status ? args.status.split(',') : null;
        const groupFilter = args.group ? args.group.split(',') : null;
        const priorityFilter = args.priority ? args.priority.split(',') : null;
        const systemFilter = args.system ? args.system.split(',') : null;
        
        console.error(`Connecting to ${args.relay}...`);
        if (systemFilter) console.error(`Filtering by system: ${args.system}`);
        
        gun.get(`${args.prefix}-issues`).map().once((data, key) => {
            if (!data || !data.id || seen.has(data.id)) return;
            seen.add(data.id);
            if (statusFilter && !statusFilter.includes(data.status)) return;
            if (groupFilter && !groupFilter.includes(data.group)) return;
            if (priorityFilter && !priorityFilter.includes(data.priority)) return;
            if (systemFilter && !systemFilter.includes(data.system)) return;
            issues.push({
                id: data.id, title: data.title, description: data.description || '',
                group: data.group || 'other', priority: data.priority || 'medium',
                status: data.status || 'open', reporter: data.reporter || 'Anonymous',
                device: data.device || 'Unknown', browser: data.browser || 'Unknown',
                system: data.system || 'default',
                createdAt: data.createdAt, updatedAt: data.updatedAt
            });
        });
        
        setTimeout(() => {
            issues.sort((a, b) => {
                const pa = CONFIG.priorities[a.priority]?.weight || 2;
                const pb = CONFIG.priorities[b.priority]?.weight || 2;
                if (pb !== pa) return pb - pa;
                return (b.createdAt || 0) - (a.createdAt || 0);
            });
            console.error(`Found ${issues.length} issues`);
            resolve(args.limit ? issues.slice(0, args.limit) : issues);
        }, args.timeout);
    });
}

function formatJSON(issues) {
    return JSON.stringify({ fetchedAt: new Date().toISOString(), count: issues.length, issues }, null, 2);
}

function formatMarkdown(issues) {
    let md = `# 待處理問題清單\n\n> 取得時間: ${new Date().toLocaleString('zh-TW')}\n> 問題數量: ${issues.length}\n\n`;
    if (issues.length === 0) return md + '_目前沒有待處理問題_\n';
    
    const grouped = {};
    issues.forEach(i => { if (!grouped[i.group]) grouped[i.group] = []; grouped[i.group].push(i); });
    
    Object.entries(CONFIG.groups).forEach(([gid, ginfo]) => {
        if (!grouped[gid]) return;
        md += `## ${ginfo.icon} ${ginfo.name} (${grouped[gid].length})\n\n`;
        grouped[gid].forEach(issue => {
            const p = CONFIG.priorities[issue.priority] || CONFIG.priorities.medium;
            const s = CONFIG.statuses[issue.status] || CONFIG.statuses.open;
            const cb = ['open', 'in-progress'].includes(issue.status) ? '[ ]' : '[x]';
            const time = issue.createdAt ? new Date(issue.createdAt).toLocaleDateString('zh-TW') : '';
            md += `- ${cb} **${issue.title}** \`${p.icon}${p.name}\` \`${s.name}\`\n`;
            if (issue.description) md += `  > ${issue.description}\n`;
            md += `  - ID: \`${issue.id}\` | 👤 ${issue.reporter} | 📱 ${issue.device} | 🕐 ${time}\n\n`;
        });
    });
    return md;
}

function formatSummary(issues) {
    let s = `=== Issue Summary (${new Date().toLocaleString('zh-TW')}) ===\n\nTotal: ${issues.length} pending issues\n\n`;
    if (issues.length === 0) return s + 'No pending issues found.\n';
    
    s += 'By Priority:\n';
    const byP = {};
    issues.forEach(i => { byP[i.priority] = (byP[i.priority] || 0) + 1; });
    Object.entries(CONFIG.priorities).forEach(([id, info]) => { if (byP[id]) s += `  ${info.icon} ${info.name}: ${byP[id]}\n`; });
    
    s += '\nBy Group:\n';
    const byG = {};
    issues.forEach(i => { byG[i.group] = (byG[i.group] || 0) + 1; });
    Object.entries(CONFIG.groups).forEach(([id, info]) => { if (byG[id]) s += `  ${info.icon} ${info.name}: ${byG[id]}\n`; });
    
    s += '\nTop Issues:\n';
    issues.slice(0, 5).forEach((i, idx) => { s += `  ${idx + 1}. [${CONFIG.priorities[i.priority]?.icon || '•'}] ${i.title}\n`; });
    if (issues.length > 5) s += `  ... and ${issues.length - 5} more\n`;
    return s;
}

async function main() {
    const args = parseArgs();
    const issues = await fetchIssues(args);
    let output;
    switch (args.format) {
        case 'markdown': output = formatMarkdown(issues); break;
        case 'summary': output = formatSummary(issues); break;
        default: output = formatJSON(issues);
    }
    if (args.output) { fs.writeFileSync(args.output, output, 'utf8'); console.error(`Saved to ${args.output}`); }
    else console.log(output);
    process.exit(0);
}

module.exports = { fetchIssues, CONFIG };
if (require.main === module) main();
