nvprof_run() {
  P_TARGET=$1
  P_COMPILER="nvcc"
  P_MINIMIG_ARGS=$2

  COLOR_CLEAR='\033[0m'
  COLOR_LIGHTGREEN='\033[1;32m'

  P_EXEC=main_"$P_TARGET"_"$P_COMPILER"
  F_OUTPUT=$P_EXEC.nvprof.txt

  rm -rf $F_OUTPUT

  TARGET=$P_TARGET COMPILER=$P_COMPILER make clean all
  echo "$F_OUTPUT"
  nvprof --metrics flop_count_sp --metrics l2_read_transactions --metrics l2_write_transactions --metrics dram_read_transactions --metrics dram_write_transactions ./"$P_EXEC" $P_MINIMIG_ARGS > "$F_OUTPUT" 2>&1
}
