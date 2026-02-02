#!/usr/bin/env node
/**
 * Batch process multiple issues from relay server
 * 
 * Usage:
 *   node batch-process.js [options]
 * 
 * Options:
 *   --relay URL         Relay server URL
 *   --prefix PREFIX     Node prefix
 *   --target PATH       Target project path
 *   --group GROUP       Filter by group
 *   --priority PRIORITY Filter by priority (default: critical,high)
 *   --limit N           Maximum issues to process
 *   --output-dir DIR    Output directory for reports
 *   --dry-run           Analyze only, don't create reports
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// Import local modules
const { fetchIssues, CONFIG } = require('./fetch-issues.js');
const { processIssue, analyzeIssue, generateTestPlan } = require('./process-issue.js');

// Parse CLI arguments
function parseArgs() {
    const args = {
        relay: process.env.ISSUE_RELAY_URL || CONFIG.relayUrl,
        prefix: process.env.ISSUE_NODE_PREFIX || CONFIG.nodePrefix,
        target: process.env.TARGET_SYSTEM_PATH || null,
        group: null,
        priority: 'critical,high',
        status: 'open,in-progress',
        limit: null,
        outputDir: './issue-reports',
        dryRun: false,
        timeout: 5000
    };
    
    const argv = process.argv.slice(2);
    for (let i = 0; i < argv.length; i++) {
        switch (argv[i]) {
            case '--relay':
                args.relay = argv[++i];
                break;
            case '--prefix':
                args.prefix = argv[++i];
                break;
            case '--target':
                args.target = argv[++i];
                break;
            case '--group':
                args.group = argv[++i];
                break;
            case '--priority':
                args.priority = argv[++i];
                break;
            case '--status':
                args.status = argv[++i];
                break;
            case '--limit':
                args.limit = parseInt(argv[++i]);
                break;
            case '--output-dir':
                args.outputDir = argv[++i];
                break;
            case '--dry-run':
                args.dryRun = true;
                break;
            case '--timeout':
                args.timeout = parseInt(argv[++i]);
                break;
            case '--help':
                showHelp();
                process.exit(0);
        }
    }
    
    return args;
}

function showHelp() {
    console.log(`
Batch process issues from Gun.js relay server

Usage: node batch-process.js [options]

Options:
  --relay URL         Relay server URL
  --prefix PREFIX     Node prefix (default: issue-tracker)
  --target PATH       Target project path for fixes
  --group GROUP       Filter by group
  --priority PRIORITY Filter by priority (default: critical,high)
  --status STATUS     Filter by status (default: open,in-progress)
  --limit N           Maximum issues to process
  --output-dir DIR    Output directory (default: ./issue-reports)
  --dry-run           Analyze only
  --timeout MS        Fetch timeout (default: 5000)
  --help              Show this help
`);
}

// Generate summary report
function generateSummary(issues, results) {
    const lines = [];
    const sep = '='.repeat(70);
    
    lines.push(sep);
    lines.push('BATCH ISSUE PROCESSING SUMMARY');
    lines.push(sep);
    lines.push(`Generated: ${new Date().toLocaleString('zh-TW')}`);
    lines.push(`Total Issues: ${issues.length}`);
    lines.push(`Processed: ${results.length}`);
    lines.push(`Successful: ${results.filter(r => r.success).length}`);
    lines.push(`Failed: ${results.filter(r => !r.success).length}`);
    lines.push('');
    
    // By Priority
    lines.push('-'.repeat(40));
    lines.push('BY PRIORITY:');
    const byPriority = {};
    issues.forEach(i => {
        byPriority[i.priority] = (byPriority[i.priority] || 0) + 1;
    });
    
    const priorityOrder = ['critical', 'high', 'medium', 'low'];
    const priorityIcons = { critical: '🔴', high: '🟠', medium: '🟡', low: '🟢' };
    priorityOrder.forEach(p => {
        if (byPriority[p]) {
            lines.push(`  ${priorityIcons[p]} ${p.toUpperCase()}: ${byPriority[p]}`);
        }
    });
    lines.push('');
    
    // By Group
    lines.push('-'.repeat(40));
    lines.push('BY GROUP:');
    const byGroup = {};
    issues.forEach(i => {
        byGroup[i.group] = (byGroup[i.group] || 0) + 1;
    });
    
    const groupIcons = {
        system: '⚙️', ui: '🎨', account: '👤',
        data: '💾', performance: '⚡', feature: '💡', other: '📋'
    };
    Object.entries(byGroup)
        .sort((a, b) => b[1] - a[1])
        .forEach(([g, count]) => {
            lines.push(`  ${groupIcons[g] || '📋'} ${g}: ${count}`);
        });
    lines.push('');
    
    // Individual Results
    lines.push('-'.repeat(40));
    lines.push('PROCESSING RESULTS:');
    lines.push('-'.repeat(40));
    
    results.forEach(result => {
        const status = result.success ? '✅' : '❌';
        lines.push('');
        lines.push(`${status} ${result.title || 'Unknown'}`);
        lines.push(`   ID: ${result.issueId}`);
        lines.push(`   Category: ${result.category || 'unknown'}`);
        lines.push(`   Priority: ${result.priority || 'medium'}`);
        if (result.reportFile) {
            lines.push(`   Report: ${result.reportFile}`);
        }
        if (result.error) {
            lines.push(`   Error: ${result.error.substring(0, 100)}`);
        }
        if (result.suggestions && result.suggestions.length > 0) {
            lines.push(`   Suggestions:`);
            result.suggestions.slice(0, 2).forEach(s => {
                lines.push(`     • ${s}`);
            });
        }
    });
    
    lines.push('');
    lines.push(sep);
    lines.push('END OF SUMMARY');
    lines.push(sep);
    
    return lines.join('\n');
}

// Process single issue and return result
function processSingleIssue(issue, outputDir, dryRun) {
    try {
        const analysis = analyzeIssue(issue);
        const testPlan = generateTestPlan(issue);
        
        let reportFile = null;
        
        if (!dryRun && outputDir) {
            const result = processIssue(issue);
            
            // Save report
            const filename = `report-${issue.id.replace(/[^a-zA-Z0-9-]/g, '_')}.txt`;
            reportFile = path.join(outputDir, filename);
            fs.writeFileSync(reportFile, result.report, 'utf8');
        }
        
        return {
            issueId: issue.id,
            title: issue.title,
            category: issue.group,
            priority: issue.priority,
            success: true,
            reportFile,
            suggestions: analysis.suggestions,
            affectedAreas: analysis.affectedAreas
        };
    } catch (error) {
        return {
            issueId: issue.id,
            title: issue.title,
            category: issue.group,
            priority: issue.priority,
            success: false,
            error: error.message
        };
    }
}

// Main
async function main() {
    const args = parseArgs();
    
    console.log('='.repeat(60));
    console.log('BATCH ISSUE PROCESSOR');
    console.log('='.repeat(60));
    console.log(`Relay: ${args.relay}`);
    console.log(`Prefix: ${args.prefix}`);
    console.log(`Priority Filter: ${args.priority}`);
    console.log(`Output Dir: ${args.outputDir}`);
    console.log(`Dry Run: ${args.dryRun}`);
    console.log('='.repeat(60));
    console.log('');
    
    // Create output directory
    if (!args.dryRun && args.outputDir) {
        fs.mkdirSync(args.outputDir, { recursive: true });
    }
    
    // Fetch issues
    console.log('Fetching issues from relay server...');
    
    const issues = await fetchIssues({
        relay: args.relay,
        prefix: args.prefix,
        group: args.group,
        priority: args.priority,
        status: args.status,
        limit: args.limit,
        timeout: args.timeout
    });
    
    if (issues.length === 0) {
        console.log('No issues found matching criteria.');
        return;
    }
    
    console.log(`Found ${issues.length} issues to process`);
    console.log('');
    
    // Show issues
    console.log('-'.repeat(40));
    console.log('ISSUES TO PROCESS:');
    console.log('-'.repeat(40));
    
    const priorityIcons = { critical: '🔴', high: '🟠', medium: '🟡', low: '🟢' };
    issues.forEach((issue, i) => {
        const icon = priorityIcons[issue.priority] || '•';
        console.log(`${i + 1}. ${icon} [${issue.group || 'other'}] ${issue.title}`);
    });
    console.log('');
    
    if (args.dryRun) {
        console.log('DRY RUN MODE - Analyzing only');
        console.log('');
    }
    
    // Process each issue
    console.log('-'.repeat(40));
    console.log('PROCESSING ISSUES:');
    console.log('-'.repeat(40));
    
    const results = [];
    for (let i = 0; i < issues.length; i++) {
        const issue = issues[i];
        console.log('');
        console.log(`[${i + 1}/${issues.length}] Processing: ${issue.title}`);
        
        const result = processSingleIssue(issue, args.outputDir, args.dryRun);
        results.push(result);
        
        if (result.success) {
            console.log(`  ✅ Completed`);
            if (result.reportFile) {
                console.log(`     Report: ${result.reportFile}`);
            }
        } else {
            console.log(`  ❌ Failed: ${result.error || 'Unknown error'}`);
        }
    }
    
    // Generate summary
    console.log('');
    console.log('='.repeat(60));
    console.log('GENERATING SUMMARY...');
    
    const summary = generateSummary(issues, results);
    
    if (!args.dryRun && args.outputDir) {
        const summaryFile = path.join(args.outputDir, 'SUMMARY.txt');
        fs.writeFileSync(summaryFile, summary, 'utf8');
        console.log(`Summary saved to: ${summaryFile}`);
        
        // Also save JSON results
        const jsonFile = path.join(args.outputDir, 'results.json');
        fs.writeFileSync(jsonFile, JSON.stringify({
            processedAt: new Date().toISOString(),
            totalIssues: issues.length,
            successful: results.filter(r => r.success).length,
            failed: results.filter(r => !r.success).length,
            results
        }, null, 2), 'utf8');
        console.log(`JSON results saved to: ${jsonFile}`);
    }
    
    console.log('');
    console.log(summary);
    
    // Exit with error code if any failed
    const failedCount = results.filter(r => !r.success).length;
    if (failedCount > 0) {
        process.exit(1);
    }
}

// Run
main().catch(err => {
    console.error(`Error: ${err.message}`);
    process.exit(1);
});
