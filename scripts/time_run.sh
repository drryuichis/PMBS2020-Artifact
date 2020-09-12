time_run() {
  P_TARGET=$1
  P_TARGET_NAME=$2
  P_COMPILER=$3
  P_COMPILER_NAME=$4
  P_REPORT=$5
  P_MINIMIG_ARGS=$6

  TARGET=$P_TARGET COMPILER=$P_COMPILER make clean all

  CPU_ARCH=`lscpu | sed -e '/^\<Model name\>/!d' -e 's/,/ /g' -e 's/[ ]\+/ /g' | cut -d ' ' -f 3`
  if [ -z "$CPU_ARCH" ]
  then
      CPU_ARCH=`uname -m`
  fi

  P_DESC="$P_COMPILER_NAME, on $CPU_ARCH, $P_TARGET_NAME"

  COLOR_CLEAR='\033[0m'
  COLOR_LIGHTGREEN='\033[1;32m'

  echo "### $P_DESC" >> $P_REPORT
  echo '```' >> $P_REPORT
  echo -e "${COLOR_LIGHTGREEN}./main_${P_TARGET}_${P_COMPILER} $P_MINIMIG_ARGS $COLOR_CLEAR"
  ./main_"$P_TARGET"_"$P_COMPILER" $P_MINIMIG_ARGS >> $P_REPORT
  echo '```' >> $P_REPORT
  echo "" >> $P_REPORT
}
