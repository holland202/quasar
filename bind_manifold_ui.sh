#!/bin/bash
echo "==> Binding Vector Manifold to the Visualizer..."

cd ~/vault-tools/galaxy || exit 1

# 1. Overwrite server.js to include the Manifold API Endpoint
cat > server.js << 'SERVER_EOF'
const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 8080;

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

const TOPOLOGY = [
  { id: "skn-v1-", name: "skn-v1-", category: "Core Engine", desc: "Sovereign Evolution Core Runtime", color: 0x22ff99, orbit: 110, speed: 0.005, size: 7, url: "https://github.com/holland202/skn-v1-" },
  { id: "slc-v12-", name: "slc-v12-", category: "Logic Core", desc: "Sovereign Logic Core", color: 0xffaa33, orbit: 160, speed: 0.0038, size: 9, url: "https://github.com/holland202/slc-v12-" },
  { id: "vault-tools", name: "vault-tools", category: "Vector Knowledge Base", desc: "Specialized Vector-Mapped Storage", color: 0x88aaff, orbit: 210, speed: 0.0028, size: 8, url: "https://github.com/holland202/vault-tools" },
  { id: "Principia-Artificialis", name: "Principia-Artificialis", category: "Mathematical Physics", desc: "Topological Geodesic Mechanics", color: 0xff66cc, orbit: 270, speed: 0.002, size: 11, url: "https://github.com/holland202/Principia-Artificialis" },
  { id: "sentinel-batadal-validation", name: "sentinel-batadal-validation", category: "Validation", desc: "Sentinel Anomaly Detection", color: 0x44ffdd, orbit: 330, speed: 0.0015, size: 6, url: "https://github.com/holland202/sentinel-batadal-validation" },
  { id: "quasar", name: "quasar", category: "High Energy Compute", desc: "Quasar Accelerator Subsystem", color: 0xffee66, orbit: 380, speed: 0.0012, size: 8, url: "https://github.com/holland202/quasar" }
];

let githubCache = null;
let lastFetch = 0;
const CACHE_TTL = 15 * 60 * 1000;

app.get('/api/repos', async (req, res) => {
  const now = Date.now();
  if (githubCache && (now - lastFetch < CACHE_TTL)) return res.json(githubCache);
  try {
    const response = await fetch('https://api.github.com/users/holland202/repos');
    const repos = await response.json();
    if (!Array.isArray(repos)) throw new Error("Rate limited");
    githubCache = TOPOLOGY.map(node => {
      const liveData = repos.find(r => r.name === node.name) || {};
      return { ...node, desc: liveData.description || node.desc, url: liveData.html_url || node.url, stars: liveData.stargazers_count || 0 };
    });
    lastFetch = now;
    res.json(githubCache);
  } catch (err) {
    res.json(TOPOLOGY);
  }
});

// NEW: Endpoint to expose the raw geometric shards
app.get('/api/vault/manifold', (req, res) => {
  const indexPath = path.join(__dirname, 'db', 'manifold_data', 'index.jsonl');
  if (fs.existsSync(indexPath)) {
    const raw = fs.readFileSync(indexPath, 'utf8').trim().split('\n');
    const data = raw.filter(l => l).map(l => JSON.parse(l)).reverse().slice(0, 10);
    res.json(data);
  } else {
    res.json([]);
  }
});

app.get('/api/vault/readme', (req, res) => {
  const target = req.query.repo;
  res.send(`*No static markdown cache found for ${target}.*`);
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`\n🌌 Sovereign Galaxy v2 active [UI Manifold Binding]`);
  console.log(`🚀 Address: http://localhost:${PORT}\n`);
});
SERVER_EOF

# 2. Patch frontend raycaster logic to display the manifold
python3 -c "
import sys
with open('public/index.html', 'r') as f:
    content = f.read()

old_logic = '''      try {
        const mdRes = await fetch(\`/api/vault/readme?repo=\${data.name}\`);
        mdContent.innerHTML = marked.parse(await mdRes.text());
      } catch(err) {
        mdContent.innerHTML = \`<p>Error loading markdown.</p>\`;
      }'''

new_logic = '''      try {
        if (data.name === 'vault-tools') {
          const mRes = await fetch('/api/vault/manifold');
          const mData = await mRes.json();
          if (mData.length > 0) {
            let ui = '<h3>Active Resonance States</h3>';
            mData.forEach(node => {
              ui += \`<div style=\"background:rgba(0,170,255,0.1); padding:8px; margin-bottom:8px; border-left:3px solid #00aaff; border-radius:4px;\">
                <code style=\"color:#44ffdd\">ID: \${node.id}</code><br>
                <span style=\"font-size:0.75rem; color:#a0c0e0\">Target: \${node.filepath}</span><br>
                <span style=\"font-size:0.75rem; color:#ffaa33\">Fidelity: \${node.fidelity} | Lock: ∂²=\${node.betti_lock}</span>
              </div>\`;
            });
            mdContent.innerHTML = ui;
          } else {
            mdContent.innerHTML = \`<p>Vector Vault is empty.</p>\`;
          }
        } else {
          const mdRes = await fetch(\`/api/vault/readme?repo=\${data.name}\`);
          mdContent.innerHTML = marked.parse(await mdRes.text());
        }
      } catch(err) {
        mdContent.innerHTML = \`<p>Error loading data stream.</p>\`;
      }'''

if old_logic in content:
    content = content.replace(old_logic, new_logic)
    with open('public/index.html', 'w') as f:
        f.write(content)
    print('==> UI patched successfully.')
else:
    print('==> Warning: UI logic mismatch, patch failed.')
"

echo "==> Restarting Node server..."
pkill -f "node server.js"
node server.js
