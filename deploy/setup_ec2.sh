#!/bin/bash
# AWS EC2 Ubuntu Setup Script for Sober Crowd Engine
# Run this script with sudo on a fresh Ubuntu Server 22.04 LTS instance.

set -e

echo "Updating system..."
apt update && apt upgrade -y

echo "Installing prerequisites (Python, Nginx)..."
apt install -y python3-pip python3-venv nginx

# Create application directory
APP_DIR="/opt/sober_crowd"
mkdir -p $APP_DIR
chown -R ubuntu:ubuntu $APP_DIR

# --- Instructions for the user ---
echo ""
echo "========================================================================="
echo "Infrastructure ready! Next steps:"
echo "1. Git clone or upload the application code into $APP_DIR"
echo "2. cd $APP_DIR"
echo "3. python3 -m venv venv"
echo "4. source venv/bin/activate"
echo "5. pip install -r requirements-prod.txt"
echo "6. Copy web_ui/.env.template to web_ui/.env and set your secure passwords"
echo "========================================================================="
echo ""

read -p "Press enter when step 6 is complete to configure Services..."

# Configure Logging
mkdir -p /var/log/sober_crowd
chown -R ubuntu:ubuntu /var/log/sober_crowd

# Setup Systemd Service for Gunicorn
echo "Setting up Gunicorn service..."
cat > /etc/systemd/system/sober_crowd.service << EOF
[Unit]
Description=Gunicorn daemon for Sober Crowd Engine
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/gunicorn --config $APP_DIR/deploy/gunicorn_config.py --chdir $APP_DIR/web_ui app:app

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl start sober_crowd
systemctl enable sober_crowd

# Configure Nginx
echo "Configuring Nginx..."
cp $APP_DIR/deploy/nginx_site.conf /etc/nginx/sites-available/sober_crowd
ln -s /etc/nginx/sites-available/sober_crowd /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Adjust Firewall
ufw allow 'Nginx Full'
ufw allow ssh
yes | ufw enable

# Restart Nginx
systemctl restart nginx

echo "Deployment complete! Application should be accessible at your Elastic IP/Domain on port 80."
