#!/usr/bin/env node
const Gun = require('gun');
const gun = Gun({ peers: ['https://relay-o.oopdoo.org.ua/gun'], localStorage: false });
const seen = new Set();
console.log('Fetching all issues from issue-tracker-issues...');
console.log('Looking for: system=issue-tracker, status=open or in-progress\n');

gun.get('issue-tracker-issues').map().once((data, key) => {
    if (!data || !data.id || seen.has(data.id)) return;
    seen.add(data.id);

    const isIssueTracker = data.system === 'issue-tracker';
    const isOpen = data.status === 'open' || data.status === 'in-progress';
    const match = isIssueTracker && isOpen ? '✅ MATCH' : '❌ SKIP';

    console.log(`${match} | system="${data.system}" | status="${data.status}" | "${data.title}"`);
});

setTimeout(() => {
    console.log('\nDone. Found', seen.size, 'total issues');
    process.exit(0);
}, 8000);
