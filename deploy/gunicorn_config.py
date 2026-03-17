import multiprocessing

bind            = "127.0.0.1:5001"
workers         = 2                         # 2 is enough — pipeline runs as subprocess
timeout         = 300
graceful_timeout= 300
accesslog       = "/var/log/crimenabi/gunicorn-access.log"
errorlog        = "/var/log/crimenabi/gunicorn-error.log"
loglevel        = "info"
proc_name       = "crimenabi_dashboard"
