#!/bin/bash 
#SBATCH --time=00:30:00
#SBATCH --nodes=1

module load python
module load likwid

SECONDS=0

IND_JOBS=$1
REP=$2
WORKDIR=~/1000genome-measurements/jobs$IND_JOBS.r$REP
RESULTDIR=$WORKDIR/1000genome-group$GROUP.jobs$IND_JOBS.r$REP

mkdir $WORKDIR
mkdir $RESULTDIR
cd $WORKDIR

cp ~/workflows/1000genome-workflow/bin/*.py $WORKDIR
cp ~/workflows/1000genome-workflow/data/data-chr1/* $WORKDIR
cp ~/workflows/1000genome-workflow/data/populations/* $WORKDIR

SECONDS=0

#chmod individuals files
chmod +x individuals.py
chmod +x individuals_merge.py
chmod +x frequency.py
chmod +x mutation_overlap.py
chmod +x sifting.py


# start individual jobs
STEPWIDTH=$((1250 / $IND_JOBS ))
PIDS_IND_JOB=( 0 )
FILES_IND=""

for (( i = 0; i < $IND_JOBS; i++ )) 
do
  CORE=$i
  START=$(( $i * $STEPWIDTH ))
  START=$(( $START + 1 ))
  END=$(( $START + STEPWIDTH ))
  if (( $i < $(($IND_JOBS / 2 )) ))
  then
    echo "i=$i, setting NUMA=0"
    NUMA=0
  else 
    echo "i=$i, setting NUMA=1"
    NUMA=1
  fi
  likwid-pin -C M$NUMA:$CORE ./individuals.py ALL.chr1.1250.vcf 1 $START $END 1250 &> $RESULTDIR/individuals_group_$GROUP_id_$i &
  pid=$!
  PIDS_IND_JOB[$i]=$pid
  FILES_IND+="chr1n-$START-$END.tar.gz "
done

for i_pid in ${PIDS_IND_JOB[@]}; do
	wait $i_pid
done

echo "individuals jobs"
date +%T -d "1/1 + $SECONDS sec"
SECONDS=0

# start sifting
likwid-perfctr -f -M 1 -g $GROUP -C 0 ./sifting.py ALL.chr1.phase3_shapeit2_mvncall_integrated_v5.20130502.sites.annotation.vcf 1 &> $RESULTDIR/sifting_group_$GROUP

echo "sifting jobs"
date +%T -d "1/1 + $SECONDS sec"
SECONDS=0


# start individuals_merge
#1 because we compute for chromosome 1. 
likwid-perfctr -f -M 1 -g $GROUP -C 0 ./individuals_merge.py 1 $FILES_IND &> $RESULTDIR/individuals_merge_group_$GROUP


echo "individuals_merge jobs"
date +%T -d "1/1 + $SECONDS sec"
SECONDS=0


# start mutations_overlap
POPULATIONS=( AFR ALL AMR EAS EUR GBR SAS )
CORE=0
PIDS_MUT_JOB=( 0 )
PIDS_FREQ_JOB=( 0 )

for population in ${POPULATIONS[@]}; do
  likwid-perfctr -f -M 1 -g $GROUP -C $CORE ./mutation_overlap.py -c 1 -pop $population &> $RESULTDIR/mutation_overlap_group_$GROUP_id_$CORE &
  pid=$!
  PIDS_MUT_JOB[$CORE]=$pid
  CORE=$(( $CORE + 1 ))
done

for i_pid in ${PIDS_MUT_JOB[@]}; do
	wait $i_pid
done

echo "mutation_overlap jobs"
date +%T -d "1/1 + $SECONDS sec"
SECONDS=0

# start frequency
CORE=0

for population in ${POPULATIONS[@]}; do
  likwid-perfctr -f -M 1 -g $GROUP -C $CORE ./frequency.py -c 1 -pop $population &> $RESULTDIR/frequency_group_$GROUP_id_$CORE &
  pid=$!
  PIDS_FREQ_JOB[$CORE]=$pid
  CORE=$(( $CORE + 1 ))
done

for i_pid in ${PIDS_FREQ_JOB[@]}; do
	wait $i_pid
done

echo "frequency jobs"
date +%T -d "1/1 + $SECONDS sec"
SECONDS=0


