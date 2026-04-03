adduser shubham
usermod -aG sudo shubham

rsync --archive --chown=shubham:shubham ~/.ssh /home/shubham


mkdir -p /home/shubham/.ssh
nano /home/shubham/.ssh/authorized_keys
chown -R shubham:shubham /home/shubham/.ssh
chmod 700 /home/shubham/.ssh
chmod 600 /home/shubham/.ssh/authorized_keys
