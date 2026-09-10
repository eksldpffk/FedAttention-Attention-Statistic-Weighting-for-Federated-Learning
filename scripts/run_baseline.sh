set -euo pipefail

python -m src.main --rounds 2 --clients-per-round 3 --local-steps 50 --strategy fedavg
python -m src.main --rounds 2 --clients-per-round 3 --local-steps 50 --strategy fedattention
