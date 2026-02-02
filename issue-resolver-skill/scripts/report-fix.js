#!/usr/bin/env node
/**
 * 問題修復回報腳本
 * 將修復結果回寫到問題回報系統
 * 
 * Usage:
 *   node report-fix.js --id ISSUE_ID --status resolved --fix-summary "修復了事件監聽器問題"
 *   node report-fix.js --id ISSUE_ID --root-cause "事件監聽器未綁定" --fix-details "移動到 DOMContentLoaded"
 * 
 * Options:
 *   --id ID              問題 ID (必填)
 *   --status STATUS      更新狀態: in-progress, resolved, closed
 *   --root-cause TEXT    根本原因描述
 *   --fix-summary TEXT   修復摘要
 *   --fix-details TEXT   修復詳細說明
 *   --files-changed TEXT 變更的檔案清單
 *   --commit TEXT        Git commit hash 或訊息
 *   --time-spent TEXT    花費時間 (如: "2h", "30m")
 *   --author TEXT        修復者名稱
 *   --relay URL          Relay server URL
 *   --prefix PREFIX      Node prefix
 */

const Gun = require('gun');
const fs = require('fs');
const path = require('path');

const CONFIG = {
    relayUrl: process.env.ISSUE_RELAY_URL || 'https://relay-o.oopdoo.org.ua/gun',
    nodePrefix: process.env.ISSUE_NODE_PREFIX || 'issue-tracker',
    validStatuses: ['open', 'in-progress', 'resolved', 'closed'],
    timeout: 10000
};

// 解析命令列參數
function parseArgs() {
    const args = {
        id: null,
        status: null,
        rootCause: null,
        fixSummary: null,
        fixDetails: null,
        filesChanged: null,
        commit: null,
        timeSpent: null,
        author: process.env.USER || 'Claude',
        relay: CONFIG.relayUrl,
        prefix: CONFIG.nodePrefix,
        reportFile: null  // 從報告檔案讀取
    };
    
    const argv = process.argv.slice(2);
    for (let i = 0; i < argv.length; i++) {
        switch (argv[i]) {
            case '--id':
                args.id = argv[++i];
                break;
            case '--status':
                args.status = argv[++i];
                break;
            case '--root-cause':
                args.rootCause = argv[++i];
                break;
            case '--fix-summary':
                args.fixSummary = argv[++i];
                break;
            case '--fix-details':
                args.fixDetails = argv[++i];
                break;
            case '--files-changed':
                args.filesChanged = argv[++i];
                break;
            case '--commit':
                args.commit = argv[++i];
                break;
            case '--time-spent':
                args.timeSpent = argv[++i];
                break;
            case '--author':
                args.author = argv[++i];
                break;
            case '--relay':
                args.relay = argv[++i];
                break;
            case '--prefix':
                args.prefix = argv[++i];
                break;
            case '--from-report':
                args.reportFile = argv[++i];
                break;
            case '--help':
            case '-h':
                showHelp();
                process.exit(0);
        }
    }
    
    return args;
}

function showHelp() {
    console.log(`
問題修復回報腳本 - 將修復結果回寫到問題回報系統

Usage:
  node report-fix.js --id ISSUE_ID --status resolved [options]
  node report-fix.js --from-report ./report.json

Options:
  --id ID              問題 ID (必填)
  --status STATUS      更新狀態: in-progress, resolved, closed
  --root-cause TEXT    根本原因描述
  --fix-summary TEXT   修復摘要 (簡短說明)
  --fix-details TEXT   修復詳細說明
  --files-changed TEXT 變更的檔案清單 (逗號分隔)
  --commit TEXT        Git commit hash 或訊息
  --time-spent TEXT    花費時間 (如: "2h", "30m")
  --author TEXT        修復者名稱 (預設: 環境變數 USER)
  --relay URL          Relay server URL
  --prefix PREFIX      Node prefix
  --from-report FILE   從 JSON 報告檔案讀取修復資訊
  --help               顯示說明

Examples:
  # 基本用法：標記為已解決並加上修復說明
  node report-fix.js --id issue-xxx --status resolved \\
    --fix-summary "修復登入按鈕無反應問題" \\
    --root-cause "事件監聽器在 DOM 載入前註冊"

  # 完整用法：包含所有詳細資訊
  node report-fix.js --id issue-xxx --status resolved \\
    --root-cause "N+1 查詢導致效能問題" \\
    --fix-summary "實作批次查詢和快取" \\
    --fix-details "1. 使用 DataLoader 批次查詢\\n2. 加入 Redis 快取" \\
    --files-changed "src/api/users.js,src/utils/cache.js" \\
    --commit "abc1234" \\
    --time-spent "3h"

  # 標記為處理中
  node report-fix.js --id issue-xxx --status in-progress \\
    --fix-summary "正在調查根本原因"
`);
}

// 建立修復評論
function createFixComment(args, issue) {
    const lines = [];
    
    // 標題
    if (args.status === 'resolved') {
        lines.push('✅ **問題已修復**');
    } else if (args.status === 'in-progress') {
        lines.push('🔧 **處理中**');
    } else if (args.status === 'closed') {
        lines.push('🔒 **問題已關閉**');
    }
    
    lines.push('');
    
    // 修復摘要
    if (args.fixSummary) {
        lines.push(`**修復摘要:** ${args.fixSummary}`);
    }
    
    // 根本原因
    if (args.rootCause) {
        lines.push(`**根本原因:** ${args.rootCause}`);
    }
    
    // 修復詳情
    if (args.fixDetails) {
        lines.push('');
        lines.push('**修復詳情:**');
        lines.push(args.fixDetails);
    }
    
    // 變更檔案
    if (args.filesChanged) {
        lines.push('');
        lines.push('**變更檔案:**');
        args.filesChanged.split(',').forEach(f => {
            lines.push(`- ${f.trim()}`);
        });
    }
    
    // Git commit
    if (args.commit) {
        lines.push('');
        lines.push(`**Commit:** \`${args.commit}\``);
    }
    
    // 花費時間
    if (args.timeSpent) {
        lines.push(`**花費時間:** ${args.timeSpent}`);
    }
    
    // 修復者和時間
    lines.push('');
    lines.push(`---`);
    lines.push(`🔧 ${args.author} | ${new Date().toLocaleString('zh-TW')}`);
    
    return {
        id: `comment-fix-${Date.now()}`,
        text: lines.join('\n'),
        author: args.author,
        createdAt: Date.now(),
        type: 'fix-report',
        fixData: {
            rootCause: args.rootCause,
            fixSummary: args.fixSummary,
            fixDetails: args.fixDetails,
            filesChanged: args.filesChanged ? args.filesChanged.split(',').map(f => f.trim()) : [],
            commit: args.commit,
            timeSpent: args.timeSpent
        }
    };
}

// 回寫修復結果到問題系統
async function reportFix(args) {
    return new Promise((resolve, reject) => {
        const gun = Gun({
            peers: [args.relay],
            localStorage: false,
            radisk: false
        });
        
        console.log(`\n🔗 連接到 ${args.relay}...`);
        
        const issuesNode = gun.get(`${args.prefix}-issues`);
        const issueNode = issuesNode.get(args.id);
        
        let resolved = false;
        
        // 取得現有問題資料
        issueNode.once((data) => {
            if (resolved) return;
            
            if (!data || !data.id) {
                resolved = true;
                reject(new Error(`找不到問題: ${args.id}`));
                return;
            }
            
            console.log(`📋 找到問題: ${data.title}`);
            console.log(`   目前狀態: ${data.status}`);
            
            // 準備更新資料
            const update = {
                updatedAt: Date.now()
            };
            
            // 更新狀態
            if (args.status) {
                update.status = args.status;
                console.log(`   新狀態: ${args.status}`);
                
                // 如果是已解決，記錄解決時間
                if (args.status === 'resolved' && data.status !== 'resolved') {
                    update.resolvedAt = Date.now();
                }
            }
            
            // 儲存修復資訊
            if (args.rootCause) {
                update.rootCause = args.rootCause;
            }
            if (args.fixSummary) {
                update.fixSummary = args.fixSummary;
            }
            if (args.fixDetails) {
                update.fixDetails = args.fixDetails;
            }
            if (args.filesChanged) {
                update.filesChanged = args.filesChanged;
            }
            if (args.commit) {
                update.commit = args.commit;
            }
            if (args.timeSpent) {
                update.timeSpent = args.timeSpent;
            }
            update.fixedBy = args.author;
            
            // 建立評論
            const comment = createFixComment(args, data);
            
            // 處理評論 (Gun.js 的陣列處理)
            let comments = [];
            try {
                if (data.comments) {
                    if (typeof data.comments === 'string') {
                        comments = JSON.parse(data.comments);
                    } else if (Array.isArray(data.comments)) {
                        comments = data.comments;
                    }
                }
            } catch (e) {
                comments = [];
            }
            comments.push(comment);
            update.comments = JSON.stringify(comments);
            
            // 寫入更新
            issueNode.put(update, (ack) => {
                if (resolved) return;
                resolved = true;
                
                if (ack.err) {
                    reject(new Error(`更新失敗: ${ack.err}`));
                    return;
                }
                
                console.log(`\n✅ 修復回報已寫入!`);
                
                // 等待同步
                setTimeout(() => {
                    resolve({
                        issueId: args.id,
                        title: data.title,
                        oldStatus: data.status,
                        newStatus: args.status || data.status,
                        fixSummary: args.fixSummary,
                        rootCause: args.rootCause,
                        updatedAt: update.updatedAt,
                        commentId: comment.id
                    });
                }, 2000);
            });
        });
        
        // 超時處理
        setTimeout(() => {
            if (!resolved) {
                resolved = true;
                reject(new Error('連線逾時'));
            }
        }, CONFIG.timeout);
    });
}

// 從報告檔案讀取
function loadFromReport(filePath) {
    if (!fs.existsSync(filePath)) {
        throw new Error(`報告檔案不存在: ${filePath}`);
    }
    
    const content = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(content);
}

// 主程式
async function main() {
    const args = parseArgs();
    
    // 從報告檔案讀取
    if (args.reportFile) {
        try {
            const report = loadFromReport(args.reportFile);
            args.id = args.id || report.issueId;
            args.status = args.status || report.status || 'resolved';
            args.rootCause = args.rootCause || report.rootCause;
            args.fixSummary = args.fixSummary || report.fixSummary;
            args.fixDetails = args.fixDetails || report.fixDetails;
            args.filesChanged = args.filesChanged || (report.filesChanged ? report.filesChanged.join(',') : null);
            args.commit = args.commit || report.commit;
            args.timeSpent = args.timeSpent || report.timeSpent;
        } catch (error) {
            console.error(`❌ 讀取報告檔案失敗: ${error.message}`);
            process.exit(1);
        }
    }
    
    // 驗證必填參數
    if (!args.id) {
        console.error('❌ 請提供問題 ID (--id)');
        process.exit(1);
    }
    
    if (args.status && !CONFIG.validStatuses.includes(args.status)) {
        console.error(`❌ 無效的狀態: ${args.status}`);
        console.error(`   有效狀態: ${CONFIG.validStatuses.join(', ')}`);
        process.exit(1);
    }
    
    // 至少需要一個修復資訊
    if (!args.status && !args.fixSummary && !args.rootCause) {
        console.error('❌ 請至少提供 --status, --fix-summary 或 --root-cause');
        process.exit(1);
    }
    
    console.log('');
    console.log('='.repeat(50));
    console.log('📝 問題修復回報');
    console.log('='.repeat(50));
    console.log(`問題 ID: ${args.id}`);
    if (args.status) console.log(`更新狀態: ${args.status}`);
    if (args.fixSummary) console.log(`修復摘要: ${args.fixSummary}`);
    if (args.rootCause) console.log(`根本原因: ${args.rootCause}`);
    console.log('='.repeat(50));
    
    try {
        const result = await reportFix(args);
        
        console.log('');
        console.log('📊 回報結果:');
        console.log(JSON.stringify(result, null, 2));
        console.log('');
        console.log('✅ 完成!');
        
        process.exit(0);
    } catch (error) {
        console.error(`\n❌ 錯誤: ${error.message}`);
        process.exit(1);
    }
}

// 匯出
module.exports = { reportFix, createFixComment, CONFIG };

// 執行
if (require.main === module) {
    main();
}
