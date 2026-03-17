#!/bin/bash
# =============================================================================
# deploy/deploy_web.sh
# =============================================================================
# Run on EC2 via SSH to set up the Crimenabi dashboard as a persistent service.
# Idempotent — safe to re-run for updates.
#
# What it does:
#   1. Pull latest code from GitHub
#   2. Install Nginx
#   3. Configure Nginx → Gunicorn reverse proxy
#   4. Create systemd service for Gunicorn
#   5. Start everything + enable on boot
# =============================================================================

set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin

WORKDIR="/home/ubuntu/crowd_management"
VENV="/opt/crimenabi_venv"
LOG_DIR="/var/log/crimenabi"
SERVICE_NAME="crimenabi"

echo ""
echo "============================================================"
echo "  Crimenabi Dashboard — Web Deployment"
echo "============================================================"

# ── 1. Pull latest code ───────────────────────────────────────────────────────
echo "[1/5] Pulling latest code from GitHub..."
cd "$WORKDIR"
git pull origin main
echo "  ✓ Code up to date"

# ── 2. Install Nginx ──────────────────────────────────────────────────────────
echo "[2/5] Installing Nginx..."
sudo apt-get install -y -q nginx
echo "  ✓ Nginx installed"

# ── 3. Configure Nginx ────────────────────────────────────────────────────────
echo "[3/5] Configuring Nginx..."
sudo cp "$WORKDIR/deploy/nginx_site.conf" /etc/nginx/sites-available/crimenabi
sudo ln -sf /etc/nginx/sites-available/crimenabi /etc/nginx/sites-enabled/crimenabi
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
echo "  ✓ Nginx configured and running"

# ── 4. Set up Gunicorn systemd service ───────────────────────────────────────
echo "[4/5] Setting up Gunicorn systemd service..."

# Log directory
sudo mkdir -p "$LOG_DIR"
sudo chown ubuntu:ubuntu "$LOG_DIR"

# Set up .env if not exists
if [ ! -f "$WORKDIR/web_ui/.env" ]; then
    cp "$WORKDIR/web_ui/.env.template" "$WORKDIR/web_ui/.env"
    # Generate a random admin password
    RAND_PASS=$(openssl rand -base64 12 | tr -dc 'a-zA-Z0-9' | head -c 16)
    sed -i "s/strong_password123/$RAND_PASS/" "$WORKDIR/web_ui/.env"
    echo ""
    echo "  ⚠️  Generated admin credentials:"
    echo "  Username : admin"
    echo "  Password : $RAND_PASS"
    echo "  (saved in $WORKDIR/web_ui/.env)"
    echo ""
fi

# Install gunicorn in venv
source "$VENV/bin/activate"
pip install gunicorn -q

# Create systemd service
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << EOF
[Unit]
Description=Crimenabi Crowd Management Dashboard
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=$WORKDIR
Environment="PATH=$VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
EnvironmentFile=$WORKDIR/web_ui/.env
ExecStart=$VENV/bin/gunicorn --config $WORKDIR/deploy/gunicorn_config.py web_ui.app:app
Restart=always
RestartSec=5
StandardOutput=append:$LOG_DIR/dashboard.log
StandardError=append:$LOG_DIR/dashboard.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"
echo "  ✓ Gunicorn service running"

# ── 5. Status check ───────────────────────────────────────────────────────────
echo "[5/5] Checking services..."
sleep 2
sudo systemctl is-active nginx      && echo "  ✓ Nginx   : active" || echo "  ✗ Nginx   : FAILED"
sudo systemctl is-active "$SERVICE_NAME" && echo "  ✓ Gunicorn: active" || echo "  ✗ Gunicorn: FAILED"

PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")

echo ""
echo "============================================================"
echo "  ✅  Dashboard is LIVE"
echo "  URL : http://$PUBLIC_IP"
echo ""
echo "  Credentials are in: $WORKDIR/web_ui/.env"
echo "  Logs: tail -f $LOG_DIR/dashboard.log"
echo "  Status: sudo systemctl status $SERVICE_NAME"
echo "============================================================"
