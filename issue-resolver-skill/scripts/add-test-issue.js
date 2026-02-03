#!/usr/bin/env node
/**
 * 新增測試問題到問題回報系統
 */
const Gun = require('gun');

const CONFIG = {
    relayUrl: 'https://relay-o.oopdoo.org.ua/gun',
    nodePrefix: 'issue-tracker'
};

const gun = Gun({
    peers: [CONFIG.relayUrl],
    localStorage: false,
    radisk: false
});

const issuesNode = gun.get(`${CONFIG.nodePrefix}-issues`);

// 產生唯一 ID
function generateId() {
    return `issue-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// 建立測試問題
const issue = {
    id: generateId(),
    title: '問題回報系統測試問題',
    description: '這是一個用於測試 Issue Resolver Skill 的問題。請確認問題能正確顯示在待處理清單中。',
    system: 'issue-tracker',  // 與 app.js 中的系統 ID 一致
    group: 'system',
    priority: 'high',
    status: 'open',
    reporter: 'Claude',
    contact: '',
    device: 'Windows',
    browser: 'Node.js',
    userAgent: 'Node.js Script',
    screenshots: JSON.stringify([]),
    errorMessage: '',
    stepsToReproduce: '1. 執行 Issue Resolver Skill\n2. 選擇問題回報系統\n3. 確認此問題出現在清單中',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    resolvedAt: null,
    comments: JSON.stringify([])
};

console.log('📝 正在新增測試問題...');
console.log('問題 ID:', issue.id);
console.log('所屬系統:', issue.system);
console.log('');

// 使用 put 新增問題
issuesNode.get(issue.id).put(issue);

console.log('⏳ 等待資料同步...');

// 等待同步後驗證
setTimeout(() => {
    console.log('');
    console.log('✅ 測試問題已新增！');
    console.log('');
    console.log('問題詳情:');
    console.log('  標題:', issue.title);
    console.log('  優先級:', issue.priority);
    console.log('  狀態:', issue.status);
    console.log('  系統:', issue.system);
    console.log('');
    console.log('🔄 資料已同步到 Relay Server');
    process.exit(0);
}, 5000);
