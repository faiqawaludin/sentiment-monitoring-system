#!/bin/bash

echo "Shutting down Remosy service"

# 'down' akan mematikan dan menghapus container (Data aman karena ada volume)
docker compose down

echo "Container is shutted down"

# Opsional: Matikan WSL total jika mau benar-benar bersih (Uncomment jika butuh)
# echo "🔴 Shutting down WSL"
# powershell.exe -Command "wsl --shutdown"
