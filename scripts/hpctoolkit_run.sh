hpctoolkit_run() {
  P_TARGET=$1
  P_COMPILER=$2
  P_FLAGS=$3
  P_HPCRUN_FLAGS=$4
  P_MINIMIG_ARGS=$5

  COLOR_CLEAR='\033[0m'
  COLOR_LIGHTGREEN='\033[1;32m'

  P_EXEC=main_"$P_TARGET"_"$P_COMPILER"
  D_MEASUREMENTS=hpctoolkit-$P_EXEC-measurements
  F_STRUCT=$P_EXEC.hpcstruct
  D_DB=hpctoolkit-$P_EXEC-database
  F_ARCHIVE=$D_DB.tar.gz

  rm -rf $D_MEASUREMENTS
  rm -rf $F_STRUCT
  rm -rf $D_DB
  rm -rf $F_ARCHIVE

  echo -e "${COLOR_LIGHTGREEN}Compiling with linemapping information ${COLOR_CLEAR}"
  HPCTOOLKITFLAGS=$P_FLAGS TARGET=$P_TARGET COMPILER=$P_COMPILER make clean all

  echo -e "${COLOR_LIGHTGREEN}hpcrun $P_HPCRUN_FLAGS -t ./$P_EXEC $P_MINIMIG_ARGS ${COLOR_CLEAR}"
  hpcrun $P_HPCRUN_FLAGS -t ./"$P_EXEC" $P_MINIMIG_ARGS

  echo -e "${COLOR_LIGHTGREEN}hpcstruct $P_EXEC ${COLOR_CLEAR}"
  hpcstruct $P_EXEC

  # When HPCRUN_FLAGS is set, we assume it's a GPU target...
  if [ -n "$P_HPCRUN_FLAGS" ]; then
    echo -e "${COLOR_LIGHTGREEN}hpcstruct --gpucfg yes $D_MEASUREMENTS ${COLOR_CLEAR}"
    hpcstruct --gpucfg yes $D_MEASUREMENTS
  fi

  echo -e "${COLOR_LIGHTGREEN}hpcprof -S $F_STRUCT -I ./+ $D_MEASUREMENTS ${COLOR_CLEAR}"
  hpcprof -S $F_STRUCT -I ./+ $D_MEASUREMENTS

  tar -czvf $F_ARCHIVE $D_DB
}
