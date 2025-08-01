#!/usr/bin/env sh

rm -rf out
mkdir out

rm -rf results
mkdir results

cp input.pdb results
cd results

source /usr/local/gromacs/bin/GMXRC

if [ "$systemType" = "protein" ]; then
    if [ "$uploadType" = "fetch" ]; then
        pdbfixer --pdbid="$pdbID" --add-atoms=all --keep-heterogens=none --add-residues --replace-nonstandard
    elif [ "$uploadType" = "upload" ]; then
        pdbfixer input.pdb --add-atoms=all --keep-heterogens=none --add-residues --replace-nonstandard
    fi
elif [ "$systemType" = "protein-ligand" ]; then
    pdbfixer input.pdb --add-atoms=all --keep-heterogens=none --add-residues --replace-nonstandard
else
    sed -E '/HETATM|HOH/d' input.pdb >output.pdb
fi

mv output.pdb input.pdb

gmx pdb2gmx -f input.pdb -ignh -water $waterModel -ff $forceField
gmx editconf -f conf.gro -o conf_box.gro -c -d 1.0 -bt $boxType
gmx solvate -cp conf_box.gro -cs spc216.gro -o conf_solv.gro -p topol.top

echo "integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000

nstlist         = 1
cutoff-scheme	= Verlet
coulombtype     = cutoff
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
" >ions.mdp

gmx grompp -f ions.mdp -c conf_solv.gro -p topol.top -o ions.tpr
echo 'SOL' | gmx genion -s ions.tpr -o conf_solv_ions.gro -p topol.top -neutral -conc "$salt"

echo "
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000

nstlist         = 1
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
" >em.mdp

gmx grompp -f em.mdp -c conf_solv_ions.gro -p topol.top -o em.tpr
gmx mdrun -nb gpu -v -deffnm em
mkdir em
mv em.* em
cd em
echo "Potential" | gmx energy -f em.edr -o postEM_Potential.xvg
gracebat -nxy postEM_Potential.xvg -printfile postEM_Potential.png -hdevice PNG
cd ..

STEPS_PER_NS=500000
nsteps_nvt=$(echo "$STEPS_PER_NS * $NVTEquilibrationTime" | bc)
nsteps_nvt_int=$(printf "%.0f" "$nsteps_nvt")
echo "NSTEPS NVT: $nsteps_nvt_int"

echo "
integrator              = md
nsteps                  = $nsteps_nvt_int
dt                      = 0.002
; Output control
nstxout-compressed      = 500
nstlog                  = 500
nstenergy               = 500
; Bonds
continuation            = no
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4
; Neighbor searching
cutoff-scheme           = Verlet
nstlist                 = 10
pbc                     = xyz
rlist                   = 1.2
; Van der Waals
vdwtype                 = cutoff
vdw-modifier            = force-switch
rvdw                    = 1.2
rvdw-switch             = 1.0
DispCorr                = no
; Electrostatics
coulombtype             = PME
rcoulomb                = 1.2
; Ewald
pme_order               = 4
fourierspacing          = 0.16
; Temperature coupling
tcoupl                  = V-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 0.1     0.1
ref_t                   = $temp $temp
; Pressure coupling
pcoupl                  = no
; Velocity generation
gen_vel                 = yes
gen_temp                = $temp
gen_seed                = -1
" >nvt.mdp

gmx grompp -f nvt.mdp -c em/em.gro -r em/em.gro -p topol.top -o nvt.tpr
gmx mdrun -nb gpu -v -deffnm nvt
mkdir nvt
mv nvt.* nvt
cd nvt
echo "Temperature" | gmx energy -f nvt.edr -o postNVT_Temperature.xvg
gracebat -nxy postNVT_Temperature.xvg -printfile postNVT_Temperature.png -hdevice PNG
cd ..

nsteps_npt=$(echo "$STEPS_PER_NS * $NPTEquilibrationTime" | bc)
nsteps_npt_int=$(printf "%.0f" "$nsteps_npt")
echo "NSTEPS NPT: $nsteps_npt_int"

echo "
integrator              = md
nsteps                  = $nsteps_npt_int
dt                      = 0.002
; Output control
nstxout-compressed      = 500
nstlog                  = 500
nstenergy               = 500
; Bonds
continuation            = yes
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4
; Neighbor searching
cutoff-scheme           = Verlet
pbc                     = xyz
nstlist                 = 20
rlist                   = 1.2
; Van der Waals
vdw-modifier            = force-switch
vdwtype                 = cutoff
rvdw-switch             = 1.0
rvdw                    = 1.2
DispCorr                = no
; Electrostatics
coulombtype             = PME
rcoulomb                = 1.2
; Ewald
pme_order               = 4
fourierspacing          = 0.16
; Temperature coupling
tcoupl                  = V-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 0.1     0.1
ref_t                   = $temp $temp
; Pressure coupling
pcoupl                  = C-rescale
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = 1.0
compressibility         = 4.5e-5
refcoord_scaling        = com
; Velocity generation
gen_vel                 = no
" >npt.mdp

gmx grompp -f npt.mdp -c nvt/nvt.gro -r nvt/nvt.gro -t nvt/nvt.cpt -p topol.top -o npt.tpr
gmx mdrun -nb gpu -v -deffnm npt
mkdir npt
mv npt.* npt
cd npt
echo "Pressure" | gmx energy -f npt.edr -o postNPT_Pressure.xvg
gracebat -nxy postNPT_Pressure.xvg -printfile postNPT_Pressure.png -hdevice PNG
echo "Density" | gmx energy -f npt.edr -o postNPT_Density.xvg
gracebat -nxy postNPT_Density.xvg -printfile postNPT_Density.png -hdevice PNG
cd ..

nsteps=$(echo "$STEPS_PER_NS * $simulationTime" | bc)
nsteps_int=$(printf "%.0f" "$nsteps")
echo "NSTEPS: $nsteps_int"

echo "
integrator              = md
nsteps                  = $nsteps_int
dt                      = 0.002

nstenergy               = 5000      ; save energies every 10.0 ps
nstlog                  = 5000      ; update log file every 10.0 ps
nstxout-compressed      = 5000      ; save coordinates every 10.0 ps
compressed-x-grps       = System

; Bonds
continuation            = yes
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4

; NEIGHBOR SEARCHING PARAMETERS
cutoff-scheme           = Verlet
rlist                   = 1.2
nstlist                 = 20
pbc                     = xyz
; Van der Waals OPTIONS
vdwtype                 = cutoff
vdw-modifier            = force-switch
rvdw                    = 1.2
rvdw-switch             = 1.0
DispCorr                = no
; ELECTROSTATICS OPTIONS
coulombtype             = PME
rcoulomb                = 1.2
; Ewald parameters
pme_order               = 4
fourierspacing          = 0.16
; Temperature coupling
tcoupl                  = V-rescale
tc-grps                 = Protein   Water_and_ions
tau_t                   = 0.1     0.1
ref_t                   = $temp $temp
; Pressure coupling
pcoupl                  = Parrinello-Rahman
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = 1.0
compressibility         = 4.5e-5
; Velocity generation
gen_vel                 = no
" >md.mdp

gmx grompp -v -f md.mdp -c npt/npt.gro -t npt/npt.cpt -p topol.top -o md.tpr
gmx mdrun -nb gpu -v -deffnm md
mkdir md
mv md.* md
cd md
echo 1 0 | gmx trjconv -s md.tpr -f md.xtc -o traj.xtc -pbc mol -center
gmx editconf -f md.gro -o first_frame.pdb
gmx check -f traj.xtc

echo "Temperature" | gmx energy -f md.edr -o postMD_Temperature.xvg
gracebat -nxy postMD_Temperature.xvg -printfile postMD_Temperature.png -hdevice PNG
echo "Pressure" | gmx energy -f md.edr -o postMD_Pressure.xvg
gracebat -nxy postMD_Pressure.xvg -printfile postMD_Pressure.png -hdevice PNG
echo "Density" | gmx energy -f md.edr -o postMD_Density.xvg
gracebat -nxy postMD_Density.xvg -printfile postMD_Density.png -hdevice PNG
echo "Total-Energy" | gmx energy -f md.edr -o postMD_TotalEnergy.xvg
gracebat -nxy postMD_TotalEnergy.xvg -printfile postMD_TotalEnergy.png -hdevice PNG
echo "Potential" | gmx energy -f md.edr -o postMD_Potential.xvg
gracebat -nxy postMD_Potential.xvg -printfile postMD_Potential.png -hdevice PNG
echo "Kinetic-En." | gmx energy -f md.edr -o postMD_KineticEnergy.xvg
gracebat -nxy postMD_KineticEnergy.xvg -printfile postMD_KineticEnergy.png -hdevice PNG

echo 4 4 | gmx rms -s md.tpr -f traj.xtc -o backbone-rmsd.xvg -tu ns
gracebat -nxy backbone-rmsd.xvg -printfile backbone-rmsd.png -hdevice PNG
echo 4 | gmx rmsf -s md.tpr -f traj.xtc -o backbone-rmsf.xvg -res
gracebat -nxy backbone-rmsf.xvg -printfile backbone-rmsf.png -hdevice PNG
echo 3 | gmx rmsf -s md.tpr -f traj.xtc -o c-alpha-rmsf.xvg -res
gracebat -nxy c-alpha-rmsf.xvg -printfile c-alpha-rmsf.png -hdevice PNG
echo "Protein" | gmx gyrate -s md.tpr -f traj.xtc -o postMD_RadiusOfGyration.xvg
gracebat -nxy postMD_RadiusOfGyration.xvg -printfile postMD_RadiusOfGyration.png -hdevice PNG
cd ..

rm mdout.mdp
rm -f ./*#*
rm ions.*

cd ..
cp -r results/* out/

# export PATH="$PATH:/root/miniconda3/bin"

# if [ "$systemType" = "protein" ]; then
#     if [ "$uploadType" = "fetch" ]; then
#         pdbfixer --pdbid="$pdbID" --output=input_clean.pdb --add-atoms=heavy --keep-heterogens=none --add-residues
#     elif [ "$uploadType" = "upload" ]; then
#         pdbfixer input.pdb --output=input_clean.pdb --add-atoms=heavy --keep-heterogens=none --add-residues
#     fi
# elif [ "$systemType" = "protein-ligand" ]; then
#     pdbfixer input.pdb --output=input_clean.pdb --add-atoms=heavy --keep-heterogens=none --add-residues
# else
#     sed -E '/HETATM|HOH/d' input.pdb > input_clean.pdb
# fi

# cp -r /home/ubuntu/CHAPERONg/save-clean-setup out
# mv input_clean.pdb /home/ubuntu/out/input_clean.pdb
# cd out

# source /usr/local/gromacs/bin/GMXRC

# echo "bt    =   $boxType
# nt  =   64
# nb  =   gpu
# auto_mode   =   full
# deffnm  =   input
# water   =  $waterModel
# ff  =   wd
# movieFrame  =   250
# conc    =   $salt
# temp    =   $temp
# inputtraj   =   $traj
# clustr_cut  =   $clusterCut
# clustr_methd    =   $clusterMethod
# frame_beginT    =   0
# dt                =      $saveFreq" > paraFile.par

# STEPS_PER_NS=500000
# nsteps=$(echo "$STEPS_PER_NS * $simulationTime" | bc)
# nsteps_int=$(printf "%.0f" "$nsteps")
# echo "NSTEPS: $nsteps_int"

# # head -n -1 md.mdp > temp.txt && mv temp.txt md.mdp
# # echo "nsteps                  = $nsteps" >> md.mdp

# echo "
# title                   = $JobName MD simulation -- CHARMM36

# ; RUN CONTROL PARAMETERS
# ; Timestep in ps
# integrator              = md        ; leap-frog integrator
# nsteps                  = $nsteps_int  ; 2 x 50000000 = 100000000 fs (100 ns)
# dt                      = 0.002     ; 2 fs

# ; OUTPUT CONTROL OPTIONS
# ; Output frequency for energies to log file and energy file
# nstenergy               = 5000      ; save energies every 10.0 ps
# nstlog                  = 5000      ; update log file every 10.0 ps
# ; Output frequency for .xtc file
# nstxout-compressed      = 5000      ; save coordinates every 10.0 ps
# compressed-x-grps       = System

# ; Bonds
# continuation            = yes       ; Restarting after NPT
# constraint_algorithm    = lincs     ; holonomic constraints
# constraints             = h-bonds   ; bonds involving H are constrained
# lincs_iter              = 1
# lincs_order             = 4

# ; NEIGHBOR SEARCHING PARAMETERS
# cutoff-scheme           = Verlet
# rlist                   = 1.2       ; nblist cut-off
# nstlist                 = 20        ; nblist update frequency
# pbc                     = xyz
# ns_type                 = grid
# ; Van der Waals OPTIONS
# vdwtype                 = cutoff
# vdw-modifier            = force-switch
# rvdw                    = 1.2
# rvdw-switch             = 1.0
# DispCorr                = no
# ; ELECTROSTATICS OPTIONS
# coulombtype             = PME
# rcoulomb                = 1.2
# ; Ewald parameters
# pme_order               = 4
# fourierspacing          = 0.16      ; grid spacing for FFT
# ; Temperature coupling
# tcoupl                  = V-rescale
# tc-grps                 = Protein   Water_and_ions
# tau_t                   = 0.1     0.1
# ref_t                   = 300     300
# ; Pressure coupling
# pcoupl                  = Parrinello-Rahman     ; Pressure coupling on in NPT
# pcoupltype              = isotropic             ; uniform scaling of box vectors
# tau_p                   = 2.0                   ; time constant (ps)
# ref_p                   = 1.0                   ; reference pressure (bar)
# compressibility         = 4.5e-5                ; isothermal compressibility of water (bar^-1)
# ; Velocity generation
# gen_vel                 = no" > md.mdp

# nsteps_npt=$(echo "$STEPS_PER_NS * $NPTEquilibrationTime" | bc)
# nsteps_npt_int=$(printf "%.0f" "$nsteps_npt")
# echo "NSTEPS NPT: $nsteps_npt_int"

# echo "
# ; Preprocessing
# title                   = NPT equilibration -- CHARMM36
# define                  = -DPOSRES
# ; Run control
# integrator              = md        ; leap-frog integrator
# nsteps                  = $nsteps_npt_int     ; 2 x 50000 = 100 ps
# dt                      = 0.002     ; 2 fs
# ; Output control
# nstxout-compressed      = 500      ; save coordinates every 1.0 ps nstenergy               = 500       ; save energies every 1.0 ps
# nstlog                  = 500       ; update log file every 1.0 ps
# nstenergy               = 500      ; save energies every 1.0 ps
# ; Bonds
# continuation            = yes       ; Restarting after NVT
# constraint_algorithm    = lincs     ; holonomic constraints
# constraints             = h-bonds   ; bonds involving H are constrained
# lincs_iter              = 1
# lincs_order             = 4
# ; Neighbor searching
# cutoff-scheme           = Verlet
# pbc                     = xyz
# ns_type                 = grid
# nstlist                 = 20
# rlist                   = 1.2
# ; Van der Waals
# vdw-modifier            = force-switch
# vdwtype                 = cutoff
# rvdw-switch             = 1.0
# rvdw                    = 1.2
# DispCorr                = no
# ; Electrostatics
# coulombtype             = PME
# rcoulomb                = 1.2
# ; Ewald
# pme_order               = 4
# fourierspacing          = 0.16      ; grid spacing for FFT
# ; Temperature coupling
# tcoupl                  = V-rescale
# tc-grps                 = Protein Non-Protein
# tau_t                   = 0.1     0.1
# ref_t                   = 300     300     ; reference temperature (K) for each group
# ; Pressure coupling
# pcoupl                  = C-rescale
# pcoupltype              = isotropic
# tau_p                   = 2.0
# ref_p                   = 1.0             ; reference pressure(bar)
# compressibility         = 4.5e-5          ; isothermal compressibility of water (bar^-1)
# refcoord_scaling        = com
# ; Velocity generation
# gen_vel                 = no" > npt.mdp

# nsteps_nvt=$(echo "$STEPS_PER_NS * $NVTEquilibrationTime" | bc)
# nsteps_nvt_int=$(printf "%.0f" "$nsteps_nvt")
# echo "NSTEPS NVT: $nsteps_nvt_int"

# echo "
# ; Preprocessing
# title                   = NVT equilibration -- CHARMM36
# define                  = -DPOSRES
# ; Run control
# integrator              = md        ; leap-frog integrator
# nsteps                  = $nsteps_nvt_int     ; 2 x 50000 = 100 ps
# dt                      = 0.002     ; 2 fs
# ; Output control
# nstxout-compressed      = 500      ; save coordinates every 1.0 ps nstenergy               = 500       ; save energies every 1.0 ps
# nstlog                  = 500       ; update log file every 1.0 ps
# nstenergy               = 500      ; save energies every 1.0 ps
# ; Bonds
# continuation            = no        ; first md run
# constraint_algorithm    = lincs     ; holonomic constraints
# constraints             = h-bonds
# lincs_iter              = 1
# lincs_order             = 4
# ; Neighbor searching
# cutoff-scheme           = Verlet
# nstlist                 = 10
# pbc                     = xyz
# ns_type                 = grid
# rlist                   = 1.2
# ; Van der Waals
# vdwtype                 = cutoff
# vdw-modifier            = force-switch
# rvdw                    = 1.2
# rvdw-switch             = 1.0
# DispCorr                = no
# ; Electrostatics
# coulombtype             = PME
# rcoulomb                = 1.2
# ; Ewald
# pme_order               = 4
# fourierspacing          = 0.16
# ; Temperature coupling
# tcoupl                  = V-rescale
# tc-grps                 = Protein Non-Protein
# tau_t                   = 0.1     0.1
# ref_t                   = 300     300     ; reference temperature (K) for each group
# ; Pressure coupling
# pcoupl                  = no
# ; Velocity generation
# gen_vel                 = yes
# gen_temp                = 300       ; temperature for Maxwell distribution
# gen_seed                = -1" > "nvt.mdp"

# echo "; Run control
# integrator      = steep
# emtol           = 1000.0
# emstep          = 0.01
# nsteps          = 50000
# ; Neighbor searching
# ns_type         = grid
# pbc             = xyz
# cutoff-scheme   = Verlet
# rlist           = 1.2
# nstlist         = 1
# ; Electrostatics
# coulombtype     = PME
# rcoulomb        = 1.2
# ; Van der Waals
# vdwtype         = cutoff
# vdw-modifier    = force-switch
# rvdw            = 1.2
# rvdw-switch     = 1.0
# DispCorr        = no" > "minim.mdp"

# echo -e "1\n1\n0\nyes\n21\n" | ./run_CHAPERONg.sh -i input_clean.pdb --paraFile paraFile.par

# echo "4 1" | gmx trjconv -s input.gro -f input.xtc -o final_parched.xtc -fit rot+trans
# echo "0" | gmx trjconv -s input_clean.pdb -f final_parched.xtc -o final_models.pdb

# awk '/^MODEL/{last=NR} NR>last' final_models.pdb
# expect automate.exp
