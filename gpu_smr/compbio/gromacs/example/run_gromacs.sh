#!/usr/bin/env bash
set -euo pipefail

# clean dirs
rm -rf out
mkdir -p results out
rm -rf results/*
cp input.pdb results/
cd results

# Pick the first installed force field
FF=$(ls -d /usr/local/share/gromacs/top/*.ff 2>/dev/null | head -n 1 | xargs -n1 basename | sed 's/\.ff$//')

if [ -z "$FF" ]; then
    echo "No force fields found in /usr/local/share/gromacs/top/"
    exit 1
fi

echo "Using force field: $FF"

# Remove waters/ligands for simplicity
sed -E '/HETATM|HOH/d' input.pdb >clean.pdb

# Basic topology/genbox
gmx pdb2gmx -f clean.pdb -o processed.gro -water tip3p -ff "$FF" -ignh
gmx editconf -f processed.gro -o boxed.gro -c -d 1.0 -bt cubic
gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top

# Add ions (neutralize)
cat >ions.mdp <<EOF
integrator  = steep
emtol       = 1000.0
nsteps      = 500
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
EOF
gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr -maxwarn 1
echo 'SOL' | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -neutral

# Energy minimization
cat >em.mdp <<EOF
integrator  = steep
emtol       = 1000.0
nsteps      = 5000
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
EOF
gmx grompp -f em.mdp -c solv_ions.gro -p topol.top -o em.tpr
gmx mdrun -nb gpu -v -deffnm em

# Long MD run — main checkpoint/restore test
# 10 million steps @ 2 fs = 20 ns; bump if you want hours-long
cat >md.mdp <<EOF
integrator      = md
nsteps          = 10000000
dt              = 0.002
nstxout-compressed = 10000
nstenergy       = 10000
nstlog          = 10000
continuation    = yes
constraint_algorithm = lincs
constraints     = h-bonds
lincs_iter      = 1
lincs_order     = 4
cutoff-scheme   = Verlet
rlist           = 1.2
vdwtype         = cutoff
rvdw            = 1.2
coulombtype     = PME
rcoulomb        = 1.2
pme_order       = 4
fourierspacing  = 0.16
tcoupl          = V-rescale
tc-grps         = System
tau_t           = 0.1
ref_t           = 300
pcoupl          = Parrinello-Rahman
pcoupltype      = isotropic
tau_p           = 2.0
ref_p           = 1.0
compressibility = 4.5e-5
gen_temp        = 300
EOF

gmx grompp -f md.mdp -c em.gro -p topol.top -o md.tpr
gmx mdrun -nb gpu -v -deffnm md

# copy results back out
cd ..
cp -r results/* out/
