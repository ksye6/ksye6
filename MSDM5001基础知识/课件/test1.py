from mpi4py import MPI
import mpi4py
import time
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

node_name = MPI.Get_processor_name() # get the name of the node


print('Hello world from process %d at %s.' % (rank, node_name))
