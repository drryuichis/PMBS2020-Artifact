ns_run() {
  P_TARGET=$1
  P_COMPILER="nvcc"
  P_MINIMIG_ARGS=$2

  COLOR_CLEAR='\033[0m'
  COLOR_LIGHTGREEN='\033[1;32m'

  P_EXEC=main_"$P_TARGET"_"$P_COMPILER"
  F_OUTPUT=$P_EXEC.ns.txt

  rm -rf $F_OUTPUT

  TARGET=$P_TARGET COMPILER=$P_COMPILER make clean all
  echo "Output to $F_OUTPUT"
  nv-nsight-cu-cli ./"$P_EXEC" $P_MINIMIG_ARGS > "$F_OUTPUT"
}
