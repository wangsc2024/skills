#!/usr/bin/env node
/**
 * Update issue status on Gun.js relay server
 * 
 * Usage:
 *   node update-status.js --id ISSUE_ID --status STATUS
 * 
 * Options:
 *   --id ID             Issue ID to update
 *   --status STATUS     New status: open, in-progress, resolved, closed
 *   --relay URL         Relay server URL
 *   --prefix PREFIX     Node prefix
 *   --comment TEXT      Add a comment with the status change
 */

const Gun = require('gun');

const CONFIG = {
    relayUrl: process.env.ISSUE_RELAY_URL || 'https://relay-o.oopdoo.org.ua/gun',
    nodePrefix: process.env.ISSUE_NODE_PREFIX || 'issue-tracker',
    validStatuses: ['open', 'in-progress', 'resolved', 'closed']
};

// Parse CLI arguments
function parseArgs() {
    const args = {
        id: null,
        status: null,
        relay: CONFIG.relayUrl,
        prefix: CONFIG.nodePrefix,
        comment: null
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
            case '--relay':
                args.relay = argv[++i];
                break;
            case '--prefix':
                args.prefix = argv[++i];
                break;
            case '--comment':
                args.comment = argv[++i];
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
Update issue status on Gun.js relay server

Usage: node update-status.js --id ISSUE_ID --status STATUS

Options:
  --id ID             Issue ID to update (required)
  --status STATUS     New status (required): open, in-progress, resolved, closed
  --relay URL         Relay server URL
  --prefix PREFIX     Node prefix (default: issue-tracker)
  --comment TEXT      Add a comment with the status change
  --help              Show this help

Examples:
  node update-status.js --id issue-123 --status resolved
  node update-status.js --id issue-123 --status in-progress --comment "Working on fix"
`);
}

// Update issue status
async function updateStatus(args) {
    return new Promise((resolve, reject) => {
        const gun = Gun({
            peers: [args.relay],
            localStorage: false,
            radisk: false
        });
        
        console.log(`Connecting to ${args.relay}...`);
        
        const issueNode = gun.get(`${args.prefix}-issues`).get(args.id);
        
        // First, verify issue exists
        issueNode.once((data) => {
            if (!data || !data.id) {
                reject(new Error(`Issue not found: ${args.id}`));
                return;
            }
            
            console.log(`Found issue: ${data.title}`);
            console.log(`Current status: ${data.status}`);
            
            // Prepare update
            const update = {
                status: args.status,
                updatedAt: Date.now()
            };
            
            // Add resolvedAt if resolved
            if (args.status === 'resolved' && data.status !== 'resolved') {
                update.resolvedAt = Date.now();
            }
            
            // Update the issue
            issueNode.put(update, (ack) => {
                if (ack.err) {
                    reject(new Error(`Update failed: ${ack.err}`));
                } else {
                    console.log(`✅ Status updated to: ${args.status}`);
                    
                    // Add comment if provided
                    if (args.comment) {
                        const comment = {
                            id: `comment-${Date.now()}`,
                            text: args.comment,
                            author: 'System',
                            createdAt: Date.now(),
                            type: 'status-change',
                            newStatus: args.status
                        };
                        
                        // Note: Gun.js doesn't handle arrays well, 
                        // so comments would need special handling in real implementation
                        console.log(`📝 Comment: ${args.comment}`);
                    }
                    
                    // Wait a bit for sync
                    setTimeout(() => {
                        resolve({
                            issueId: args.id,
                            oldStatus: data.status,
                            newStatus: args.status,
                            updatedAt: update.updatedAt
                        });
                    }, 1000);
                }
            });
        });
        
        // Timeout
        setTimeout(() => {
            reject(new Error('Timeout: Could not connect to relay server'));
        }, 10000);
    });
}

// Main
async function main() {
    const args = parseArgs();
    
    // Validate required arguments
    if (!args.id) {
        console.error('Error: --id is required');
        process.exit(1);
    }
    
    if (!args.status) {
        console.error('Error: --status is required');
        process.exit(1);
    }
    
    if (!CONFIG.validStatuses.includes(args.status)) {
        console.error(`Error: Invalid status "${args.status}"`);
        console.error(`Valid statuses: ${CONFIG.validStatuses.join(', ')}`);
        process.exit(1);
    }
    
    try {
        const result = await updateStatus(args);
        console.log('');
        console.log('Update successful:');
        console.log(JSON.stringify(result, null, 2));
        process.exit(0);
    } catch (error) {
        console.error(`Error: ${error.message}`);
        process.exit(1);
    }
}

// Export
module.exports = { updateStatus, CONFIG };

// Run
if (require.main === module) {
    main();
}
