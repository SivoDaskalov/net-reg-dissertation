#!/bin/bash

## the p=33 setups

#setup1-small
for sim in {1..100}; do
  echo "sim=$sim" >> setup-small1.txt
  R CMD BATCH --no-save --no-restore "--args small=1 setup=1 sim=$sim" dataGen.R
  matlab -nosplash -nodesktop -nodisplay -r "small=1;gamma=Inf;outfilename='setup-small1.txt';AlgVsolve" 
done

#setup2-small
for sim in {1..100}; do
  echo "sim=$sim" >> setup-small2.txt
  R CMD BATCH --no-save --no-restore "--args small=1 setup=2 sim=$sim" dataGen.R
  matlab -nosplash -nodesktop -nodisplay -r "small=1;gamma=Inf;outfilename='setup-small2.txt';AlgVsolve" 
done

#setup3-small
for sim in {1..100}; do
  echo "sim=$sim" >> setup-small3.txt
  R CMD BATCH --no-save --no-restore "--args small=1 setup=3 sim=$sim" dataGen.R
  matlab -nosplash -nodesktop -nodisplay -r "small=1;gamma=Inf;outfilename='setup-small3.txt';AlgVsolve" 
done

#setup4-small
for sim in {1..100}; do
  echo "sim=$sim" >> setup-small4.txt
  R CMD BATCH --no-save --no-restore "--args small=1 setup=4 sim=$sim" dataGen.R
  matlab -nosplash -nodesktop -nodisplay -r "small=1;gamma=Inf;outfilename='setup-small4.txt';AlgVsolve" 
done

#setup5-small
for sim in {1..100}; do
  echo "sim=$sim" >> setup-small5.txt
  R CMD BATCH --no-save --no-restore "--args small=1 setup=5 sim=$sim" dataGen.R
  matlab -nosplash -nodesktop -nodisplay -r "small=1;gamma=Inf;outfilename='setup-small5.txt';AlgVsolve" 
done

## the p=110 setups

#setup1-big
for sim in {1..100}; do
  echo "sim=$sim" >> setup-big1.txt
  R CMD BATCH --no-save --no-restore "--args small=0 setup=1 sim=$sim" dataGen.R
  matlab -nosplash -nodesktop -nodisplay -r "small=0;gamma=Inf;outfilename='setup-big1.txt';AlgVsolve" 
done

#setup2-big
for sim in {1..100}; do
  echo "sim=$sim" >> setup-big2.txt
  R CMD BATCH --no-save --no-restore "--args small=0 setup=2 sim=$sim" dataGen.R
  matlab -nosplash -nodesktop -nodisplay -r "small=0;gamma=Inf;outfilename='setup-big2.txt';AlgVsolve" 
done

#setup3-big
for sim in {1..100}; do
  echo "sim=$sim" >> setup-big3.txt
  R CMD BATCH --no-save --no-restore "--args small=0 setup=3 sim=$sim" dataGen.R
  matlab -nosplash -nodesktop -nodisplay -r "small=0;gamma=Inf;outfilename='setup-big3.txt';AlgVsolve" 
done

#setup4-big
for sim in {1..100}; do
  echo "sim=$sim" >> setup-big4.txt
  R CMD BATCH --no-save --no-restore "--args small=0 setup=4 sim=$sim" dataGen.R
  matlab -nosplash -nodesktop -nodisplay -r "small=0;gamma=Inf;outfilename='setup-big4.txt';AlgVsolve" 
done

#setup5-big
for sim in {1..100}; do
  echo "sim=$sim" >> setup-big5.txt
  R CMD BATCH --no-save --no-restore "--args small=0 setup=5 sim=$sim" dataGen.R
  matlab -nosplash -nodesktop -nodisplay -r "small=0;gamma=Inf;outfilename='setup-big5.txt';AlgVsolve" 
done
