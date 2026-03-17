# gunicorn_config.py
import multiprocessing

# Bind to localhost on port 5001 (Nginx will proxy to this)
bind = '127.0.0.1:5001'

# Worker configuration
# Since our app launches background subprocesses that are CPU-bound, 
# we should configure Gunicorn with sufficient sync workers.
workers = multiprocessing.cpu_count() * 2 + 1

# Timeouts
# Background jobs (like pipeline) are started eagerly, but if any
# specific endpoint blocks, allow up to 120 seconds.
timeout = 120
graceful_timeout = 120

# Logging
accesslog = '/var/log/sober_crowd/gunicorn-access.log'
errorlog = '/var/log/sober_crowd/gunicorn-error.log'
loglevel = 'info'

# Process name
proc_name = 'sober_crowd_engine'
