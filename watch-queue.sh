squeue | grep "(Priority)" | awk -v user="qd36zu" '$0 ~ user {print NR; exit}'
