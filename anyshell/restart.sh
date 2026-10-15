# 加载配置文件
supervisord -c /etc/supervisor/supervisord.conf

sleep 3

# 重新启动所有服务
supervisorctl restart all