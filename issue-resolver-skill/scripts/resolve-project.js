#!/usr/bin/env node
/**
 * 專案問題解決器
 * 指定專案，下載該專案的問題，逐一診斷並修復
 * 
 * Usage:
 *   node resolve-project.js --project form-system
 *   node resolve-project.js --project document-system --priority critical
 *   node resolve-project.js --list  # 列出所有專案
 * 
 * Options:
 *   --project ID        專案 ID (必填)
 *   --priority LEVEL    優先級篩選 (default: critical,high)
 *   --group GROUP       問題群組篩選
 *   --limit N           最多處理幾個問題
 *   --dry-run           僅分析，不實際修改
 *   --output-dir DIR    報告輸出目錄
 *   --list              列出所有可用專案
 *   --config FILE       專案配置檔路徑
 */

const fs = require('fs');
const path = require('path');
// 載入配置
function loadConfig(configPath) {
    const defaultPath = path.join(__dirname, '..', 'projects.json');
    const filePath = configPath || defaultPath;
    
    if (!fs.existsSync(filePath)) {
        console.error(`配置檔不存在: ${filePath}`);
        console.error('請建立 projects.json 或使用 --config 指定配置檔');
        process.exit(1);
    }
    
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

// 解析命令列參數
function parseArgs() {
    const args = {
        project: null,
        priority: 'critical,high',
        group: null,
        limit: null,
        dryRun: false,
        outputDir: './issue-reports',
        list: false,
        config: null,
        auto: false
    };
    
    const argv = process.argv.slice(2);
    for (let i = 0; i < argv.length; i++) {
        switch (argv[i]) {
            case '--project':
            case '-p':
                args.project = argv[++i];
                break;
            case '--priority':
                args.priority = argv[++i];
                break;
            case '--group':
                args.group = argv[++i];
                break;
            case '--limit':
                args.limit = parseInt(argv[++i]);
                break;
            case '--dry-run':
                args.dryRun = true;
                break;
            case '--output-dir':
                args.outputDir = argv[++i];
                break;
            case '--list':
            case '-l':
                args.list = true;
                break;
            case '--config':
                args.config = argv[++i];
                break;
            case '--auto':
            case '-y':
                args.auto = true;
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
專案問題解決器 - 指定專案並自動處理其問題

Usage:
  node resolve-project.js --project <project-id> [options]
  node resolve-project.js --list

Options:
  -p, --project ID      專案 ID (必填，除非使用 --list)
  --priority LEVEL      優先級篩選 (default: critical,high)
  --group GROUP         問題群組篩選
  --limit N             最多處理幾個問題
  --dry-run             僅分析，不實際修改程式碼
  --output-dir DIR      報告輸出目錄 (default: ./issue-reports)
  -l, --list            列出所有可用專案
  --config FILE         專案配置檔路徑
  -y, --auto            自動模式，不詢問確認
  -h, --help            顯示說明

Examples:
  # 列出所有專案
  node resolve-project.js --list

  # 處理表單系統的緊急問題
  node resolve-project.js --project form-system --priority critical

  # 處理公文系統的所有問題 (僅分析)
  node resolve-project.js --project document-system --dry-run

  # 處理人事系統的 UI 問題
  node resolve-project.js --project hr-system --group ui
`);
}

// 列出所有專案
function listProjects(config) {
    console.log('\n📋 可用的專案:\n');
    console.log('ID                  名稱              技術棧');
    console.log('-'.repeat(60));
    
    Object.entries(config.projects).forEach(([id, proj]) => {
        const tech = proj.techStack?.join(', ') || '-';
        console.log(`${id.padEnd(20)}${proj.name.padEnd(18)}${tech}`);
    });
    
    console.log('\n使用方式: node resolve-project.js --project <ID>\n');
}

// 從 Relay 取得指定專案的問題
async function fetchProjectIssues(config, projectId, filters) {
    const Gun = require('gun');
    
    return new Promise((resolve) => {
        const gun = Gun({
            peers: [config.relay.url],
            localStorage: false,
            radisk: false
        });
        
        const issues = [];
        const seen = new Set();
        
        const statusFilter = ['open', 'in-progress'];
        const priorityFilter = filters.priority ? filters.priority.split(',') : null;
        const groupFilter = filters.group ? filters.group.split(',') : null;
        
        console.log(`\n🔍 正在從 Relay 取得 "${projectId}" 的問題...`);
        
        gun.get(`${config.relay.prefix}-issues`).map().once((data, key) => {
            if (!data || !data.id || seen.has(data.id)) return;
            seen.add(data.id);
            
            // 篩選：必須是指定專案
            if (data.system !== projectId) return;
            
            // 篩選：狀態
            if (!statusFilter.includes(data.status)) return;
            
            // 篩選：優先級
            if (priorityFilter && !priorityFilter.includes(data.priority)) return;
            
            // 篩選：群組
            if (groupFilter && !groupFilter.includes(data.group)) return;
            
            issues.push({
                id: data.id,
                title: data.title,
                description: data.description || '',
                system: data.system,
                group: data.group || 'other',
                priority: data.priority || 'medium',
                status: data.status || 'open',
                reporter: data.reporter || 'Anonymous',
                device: data.device || 'Unknown',
                browser: data.browser || 'Unknown',
                createdAt: data.createdAt,
                updatedAt: data.updatedAt
            });
        });
        
        setTimeout(() => {
            // 依優先級排序
            const priorityWeight = { critical: 4, high: 3, medium: 2, low: 1 };
            issues.sort((a, b) => {
                const pa = priorityWeight[a.priority] || 2;
                const pb = priorityWeight[b.priority] || 2;
                if (pb !== pa) return pb - pa;
                return (b.createdAt || 0) - (a.createdAt || 0);
            });
            
            resolve(filters.limit ? issues.slice(0, filters.limit) : issues);
        }, 5000);
    });
}

// 診斷單一問題
function diagnoseIssue(issue, project) {
    const TEST_PLANS = {
        system: { name: '系統問題診斷', focus: ['logs', 'config', 'dependencies'] },
        ui: { name: 'UI 問題診斷', focus: ['console', 'css', 'dom', 'events'] },
        account: { name: '帳號問題診斷', focus: ['auth', 'session', 'permissions'] },
        data: { name: '資料問題診斷', focus: ['database', 'api', 'validation'] },
        performance: { name: '效能問題診斷', focus: ['profiling', 'queries', 'caching'] },
        feature: { name: '功能需求分析', focus: ['requirements', 'design', 'impact'] }
    };
    
    const plan = TEST_PLANS[issue.group] || TEST_PLANS.system;
    
    // 根據專案技術棧提供具體建議
    const techSuggestions = [];
    if (project.techStack?.includes('react')) {
        techSuggestions.push('檢查 React 元件狀態和生命週期');
        techSuggestions.push('使用 React DevTools 檢查元件樹');
    }
    if (project.techStack?.includes('vue')) {
        techSuggestions.push('檢查 Vue 元件資料綁定');
        techSuggestions.push('使用 Vue DevTools 除錯');
    }
    if (project.techStack?.includes('node') || project.techStack?.includes('express')) {
        techSuggestions.push('檢查 Node.js 錯誤日誌');
        techSuggestions.push('使用 debug 模組追蹤請求');
    }
    if (project.techStack?.includes('mongodb')) {
        techSuggestions.push('檢查 MongoDB 查詢效能');
        techSuggestions.push('確認索引是否正確建立');
    }
    if (project.techStack?.includes('postgresql')) {
        techSuggestions.push('使用 EXPLAIN ANALYZE 檢查查詢');
        techSuggestions.push('確認資料庫連線池設定');
    }
    
    return {
        issue,
        project: project.name,
        projectPath: project.path,
        testPlan: plan,
        techSuggestions,
        entryPoints: project.entryPoints,
        suggestedFiles: getSuggestedFiles(issue, project)
    };
}

// 根據問題類型建議檢查的檔案
function getSuggestedFiles(issue, project) {
    const files = [];
    const ep = project.entryPoints || {};
    
    switch (issue.group) {
        case 'ui':
            if (ep.frontend) files.push(`${ep.frontend}/**/*.{jsx,vue,tsx,css,scss}`);
            break;
        case 'system':
        case 'performance':
            if (ep.backend) files.push(`${ep.backend}/**/*.{js,ts,py}`);
            break;
        case 'data':
            if (ep.api) files.push(`${ep.api}/**/*.{js,ts,py}`);
            if (ep.backend) files.push(`${ep.backend}/models/**/*`);
            break;
        case 'account':
            if (ep.backend) files.push(`${ep.backend}/auth/**/*`);
            if (ep.api) files.push(`${ep.api}/auth/**/*`);
            break;
    }
    
    // 通用檔案
    files.push('package.json', 'config/**/*', '.env*');
    
    return files;
}

// 產生診斷報告
function generateDiagnosisReport(diagnosis) {
    const { issue, project, projectPath, testPlan, techSuggestions, entryPoints, suggestedFiles } = diagnosis;
    
    let report = '';
    report += '='.repeat(70) + '\n';
    report += `問題診斷報告\n`;
    report += '='.repeat(70) + '\n\n';
    
    report += `【專案資訊】\n`;
    report += `專案名稱: ${project}\n`;
    report += `專案路徑: ${projectPath}\n\n`;
    
    report += `【問題資訊】\n`;
    report += `ID: ${issue.id}\n`;
    report += `標題: ${issue.title}\n`;
    report += `類型: ${issue.group}\n`;
    report += `優先級: ${issue.priority}\n`;
    report += `回報者: ${issue.reporter}\n`;
    report += `描述: ${issue.description || '無'}\n\n`;
    
    report += `【診斷計畫】\n`;
    report += `計畫: ${testPlan.name}\n`;
    report += `重點檢查:\n`;
    testPlan.focus.forEach(f => { report += `  • ${f}\n`; });
    report += '\n';
    
    report += `【技術建議】\n`;
    techSuggestions.forEach(s => { report += `  • ${s}\n`; });
    report += '\n';
    
    report += `【建議檢查檔案】\n`;
    suggestedFiles.forEach(f => { report += `  • ${f}\n`; });
    report += '\n';
    
    if (entryPoints) {
        report += `【程式進入點】\n`;
        Object.entries(entryPoints).forEach(([k, v]) => {
            report += `  ${k}: ${v}\n`;
        });
        report += '\n';
    }
    
    report += `【行動項目】\n`;
    report += `[ ] 1. 複製問題描述，在本地環境重現\n`;
    report += `[ ] 2. 根據診斷計畫檢查相關程式碼\n`;
    report += `[ ] 3. 找出根本原因\n`;
    report += `[ ] 4. 實作修復\n`;
    report += `[ ] 5. 撰寫或更新測試\n`;
    report += `[ ] 6. 驗證修復有效\n`;
    report += `[ ] 7. 提交變更並更新問題狀態\n\n`;
    
    report += `【根本原因】\n`;
    report += `(調查後填寫)\n\n`;
    
    report += `【修復方案】\n`;
    report += `(實作後填寫)\n\n`;
    
    report += '='.repeat(70) + '\n';
    report += `報告產生時間: ${new Date().toLocaleString('zh-TW')}\n`;
    report += '='.repeat(70) + '\n';
    
    return report;
}

// 主程式
async function main() {
    const args = parseArgs();
    const config = loadConfig(args.config);
    
    // 列出專案
    if (args.list) {
        listProjects(config);
        return;
    }
    
    // 檢查專案 ID
    if (!args.project) {
        console.error('❌ 請指定專案 ID，使用 --project <ID>');
        console.error('   使用 --list 查看所有可用專案');
        process.exit(1);
    }
    
    const project = config.projects[args.project];
    if (!project) {
        console.error(`❌ 找不到專案: ${args.project}`);
        console.error('   使用 --list 查看所有可用專案');
        process.exit(1);
    }
    
    console.log('\n' + '='.repeat(60));
    console.log(`🔧 專案問題解決器`);
    console.log('='.repeat(60));
    console.log(`專案: ${project.name} (${args.project})`);
    console.log(`路徑: ${project.path}`);
    console.log(`技術: ${project.techStack?.join(', ') || '-'}`);
    console.log(`優先級篩選: ${args.priority}`);
    console.log(`乾跑模式: ${args.dryRun ? '是' : '否'}`);
    console.log('='.repeat(60));
    
    // 取得問題
    const issues = await fetchProjectIssues(config, args.project, {
        priority: args.priority,
        group: args.group,
        limit: args.limit
    });
    
    if (issues.length === 0) {
        console.log('\n✅ 太好了！這個專案目前沒有待處理的問題。\n');
        return;
    }
    
    console.log(`\n📋 找到 ${issues.length} 個待處理問題:\n`);
    
    const priorityIcons = { critical: '🔴', high: '🟠', medium: '🟡', low: '🟢' };
    issues.forEach((issue, i) => {
        const icon = priorityIcons[issue.priority] || '•';
        console.log(`  ${i + 1}. ${icon} [${issue.group}] ${issue.title}`);
    });
    
    // 建立輸出目錄
    const outputDir = path.join(args.outputDir, args.project);
    if (!args.dryRun) {
        fs.mkdirSync(outputDir, { recursive: true });
    }
    
    console.log('\n' + '-'.repeat(60));
    console.log('開始處理問題...');
    console.log('-'.repeat(60));
    
    // 處理每個問題
    const results = [];
    for (let i = 0; i < issues.length; i++) {
        const issue = issues[i];
        console.log(`\n[${i + 1}/${issues.length}] 處理: ${issue.title}`);
        console.log(`  優先級: ${issue.priority.toUpperCase()}`);
        console.log(`  類型: ${issue.group}`);
        
        // 診斷
        const diagnosis = diagnoseIssue(issue, project);
        console.log(`  診斷計畫: ${diagnosis.testPlan.name}`);
        
        // 產生報告
        const report = generateDiagnosisReport(diagnosis);
        
        if (!args.dryRun) {
            const reportFile = path.join(outputDir, `${issue.id}.txt`);
            fs.writeFileSync(reportFile, report, 'utf8');
            console.log(`  📄 報告: ${reportFile}`);
        }
        
        results.push({
            issueId: issue.id,
            title: issue.title,
            priority: issue.priority,
            group: issue.group,
            diagnosis: diagnosis.testPlan.name,
            suggestions: diagnosis.techSuggestions.slice(0, 2)
        });
        
        console.log(`  ✅ 完成`);
    }
    
    // 產生摘要
    console.log('\n' + '='.repeat(60));
    console.log('📊 處理摘要');
    console.log('='.repeat(60));
    console.log(`專案: ${project.name}`);
    console.log(`處理問題數: ${results.length}`);
    console.log(`輸出目錄: ${outputDir}`);
    
    if (!args.dryRun) {
        // 儲存摘要 JSON
        const summaryFile = path.join(outputDir, 'summary.json');
        fs.writeFileSync(summaryFile, JSON.stringify({
            project: args.project,
            projectName: project.name,
            processedAt: new Date().toISOString(),
            issueCount: results.length,
            results
        }, null, 2), 'utf8');
        console.log(`\n📋 摘要檔案: ${summaryFile}`);
    }
    
    console.log('\n✅ 處理完成！\n');
    console.log('下一步:');
    console.log(`  1. 查看報告: cat ${outputDir}/<issue-id>.txt`);
    console.log(`  2. 切換到專案: cd ${project.path}`);
    console.log(`  3. 根據報告進行修復`);
    console.log(`  4. 完成後更新狀態: node scripts/update-status.js --id <issue-id> --status resolved\n`);
}

// 執行
main().catch(err => {
    console.error(`\n❌ 錯誤: ${err.message}\n`);
    process.exit(1);
});
