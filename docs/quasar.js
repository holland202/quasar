/**
 * QUASAR Web Demo — Minimal Bloch sphere visualization
 * Pure JS, no dependencies. Computes Bures distance and channel
 * deformation client-side.
 */

const canvas = document.getElementById('bloch-canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const physEl = document.getElementById('physicality');
const buresEl = document.getElementById('bures-path');

let w = 0.5, g = 0.05, steps = 10;
let isLearning = false;

function projectBloch(r) {
    const mag = Math.sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    if (mag > 1) return [r[0]/mag, r[1]/mag, r[2]/mag];
    return r;
}

function buresFidelity(r1, r2) {
    const dot = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2];
    const m1 = Math.sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2]);
    const m2 = Math.sqrt(r2[0]*r2[0] + r2[1]*r2[1] + r2[2]*r2[2]);
    const term = Math.sqrt(Math.max(0, (1-m1*m1)*(1-m2*m2)));
    return Math.max(0, Math.min(1, 0.5*(1 + dot + term)));
}

function buresDistance(r1, r2) {
    return Math.acos(Math.sqrt(buresFidelity(r1, r2)));
}

function rotate(axis, angle, r) {
    const a = [axis[0], axis[1], axis[2]];
    const mag = Math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    if (mag < 1e-9) return r;
    a[0] /= mag; a[1] /= mag; a[2] /= mag;
    const c = Math.cos(angle), s = Math.sin(angle);
    const t = 1 - c;
    const rx = (t*a[0]*a[0] + c)*r[0] + (t*a[0]*a[1] - s*a[2])*r[1] + (t*a[0]*a[2] + s*a[1])*r[2];
    const ry = (t*a[0]*a[1] + s*a[2])*r[0] + (t*a[1]*a[1] + c)*r[1] + (t*a[1]*a[2] - s*a[0])*r[2];
    const rz = (t*a[0]*a[2] - s*a[1])*r[0] + (t*a[1]*a[2] + s*a[0])*r[1] + (t*a[2]*a[2] + c)*r[2];
    return [rx, ry, rz];
}

function generateTrajectory() {
    const axis = [1, 0.5, 0.3];
    const traj = [];
    let r = [0, 0, 1];
    for (let i = 0; i < steps; i++) {
        r = rotate(axis, w, r);
        r = [r[0]*Math.exp(-g), r[1]*Math.exp(-g), r[2]*Math.exp(-g)];
        r = projectBloch(r);
        traj.push([...r]);
    }
    return traj;
}

function computePathLength(traj) {
    let total = 0;
    for (let i = 0; i < traj.length - 1; i++) {
        total += buresDistance(traj[i], traj[i+1]);
    }
    return total;
}

function draw() {
    const width = canvas.width;
    const height = canvas.height;
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, width, height);
    
    const cx = width / 2, cy = height / 2;
    const scale = 120;
    
    // Draw Bloch sphere wireframe
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, scale, 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw axes
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.beginPath();
    ctx.moveTo(cx - scale, cy); ctx.lineTo(cx + scale, cy);
    ctx.moveTo(cx, cy - scale); ctx.lineTo(cx, cy + scale);
    ctx.stroke();
    
    const traj = generateTrajectory();
    const pathLen = computePathLength(traj);
    
    // Check physicality
    let physical = true;
    for (const r of traj) {
        const mag = Math.sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        if (mag > 1.0001) physical = false;
    }
    
    physEl.textContent = physical ? 'VALID' : 'VIOLATED';
    physEl.className = physical ? 'valid' : 'invalid';
    buresEl.textContent = pathLen.toFixed(3);
    
    // Draw trajectory
    ctx.strokeStyle = isLearning ? '#22ff99' : '#ffee66';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < traj.length; i++) {
        const r = traj[i];
        const x = cx + r[0] * scale;
        const y = cy - r[2] * scale; // z is up in Bloch sphere
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    // Draw points
    for (let i = 0; i < traj.length; i++) {
        const r = traj[i];
        const x = cx + r[0] * scale;
        const y = cy - r[2] * scale;
        ctx.fillStyle = i === 0 ? '#ffffff' : (isLearning ? '#22ff99' : '#ffaa33');
        ctx.beginPath();
        ctx.arc(x, y, i === 0 ? 5 : 3, 0, Math.PI * 2);
        ctx.fill();
    }
    
    requestAnimationFrame(draw);
}

document.getElementById('w').addEventListener('input', (e) => {
    w = parseFloat(e.target.value);
    document.getElementById('val-w').textContent = w.toFixed(2);
});

document.getElementById('g').addEventListener('input', (e) => {
    g = parseFloat(e.target.value);
    document.getElementById('val-g').textContent = g.toFixed(2);
});

document.getElementById('t').addEventListener('input', (e) => {
    steps = parseInt(e.target.value);
    document.getElementById('val-t').textContent = steps;
});

document.getElementById('learn-btn').addEventListener('click', () => {
    isLearning = !isLearning;
    document.getElementById('learn-btn').textContent = 
        isLearning ? 'Reset' : 'Learn from Measurements';
});

draw();
