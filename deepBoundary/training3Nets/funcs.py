import generation_deepBoundary_lib as gdb
def square(x):
    return x**2

def getBlockCorrelation_new(k, mapblock, partitions, dotProduct, radFile, radSolution):
    i , j = mapblock[k]
    print(i,j)
    if(i==j):
        return gdb.getBlockCorrelation_sym(partitions[i],partitions[i+1], dotProduct, radFile, radSolution) 
    else:
        return gdb.getBlockCorrelation_nosym(partitions[i],partitions[i+1], partitions[j],partitions[j+1], dotProduct, radFile, radSolution) 