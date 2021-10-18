gnome-terminal -- python -m pt_server -r 100
sleep 3
gnome-terminal -- python -m pt_client -s alice
gnome-terminal -- python -m pt_client -s bob
