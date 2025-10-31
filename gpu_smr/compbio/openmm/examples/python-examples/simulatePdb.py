from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

pdb = PDBFile('input.pdb')
forcefield = ForceField('amber19-all.xml', 'amber19/tip3pfb.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.reporters.append(DCDReporter('/tmp/out/output.dcd', 10000))
simulation.reporters.append(StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True))
simulation.step(100000)
